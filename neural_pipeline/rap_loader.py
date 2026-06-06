"""RAP (Register Any Point) multi-view registration loader.

RAP uses flow matching to directly generate registered point clouds in a
canonical coordinate frame, then recovers per-view rigid transformations
via Procrustes (SVD). No MinkowskiEngine required -- pure PyTorch
transformer + lightweight local feature extractor.

Reference: Pan et al., "Register Any Point: Scaling 3D Point Cloud
Registration by Flow Matching", NeurIPS 2025.
https://github.com/PRBonn/RAP
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import numpy as np

LOG = logging.getLogger("stone3d_neural.rap")


def load_rap(
    rap_dir: str,
    weights: str,
    device: str = "cuda",
    sampling_steps: int = 10,
    rigidity_forcing: bool = True,
) -> "RAPCallable":
    """Load the RAP model for multi-view registration.

    Parameters
    ----------
    rap_dir : str
        Path to the cloned RAP repository root (contains ``rectified_point_flow/``).
    weights : str
        Path to the RAP checkpoint file (``rap_model.ckpt``).
    device : str
        Torch device string.
    sampling_steps : int
        Number of ODE solver steps for flow matching (more = slower but
        higher quality; 10 is the default used in the paper).
    rigidity_forcing : bool
        Whether to enforce per-part rigidity during flow sampling.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch is required for RAP") from e

    rap_dir = os.path.abspath(rap_dir)
    if rap_dir not in sys.path:
        sys.path.insert(0, rap_dir)
    dataset_process_dir = os.path.join(rap_dir, "dataset_process")
    if dataset_process_dir not in sys.path:
        sys.path.insert(0, dataset_process_dir)

    try:
        import hydra
        from omegaconf import OmegaConf
        from rectified_point_flow.utils import load_checkpoint_for_module
    except ImportError as e:
        raise RuntimeError(
            f"RAP imports failed. Ensure the RAP repo is at {rap_dir} "
            f"and its dependencies are installed: {e}"
        ) from e

    config_dir = os.path.join(rap_dir, "config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = hydra.compose(config_name="RAP_base")

    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, weights)
    model.eval()
    model.to(device)

    LOG.info("Loaded RAP from %s on %s (steps=%d, rigidity=%s)",
             weights, device, sampling_steps, rigidity_forcing)

    return RAPCallable(
        model=model,
        cfg=cfg,
        rap_dir=rap_dir,
        device=device,
        sampling_steps=sampling_steps,
        rigidity_forcing=rigidity_forcing,
    )


class RAPCallable:
    """Wraps the RAP model for use inside the stone reconstruction pipeline.

    Accepts a list of Open3D point clouds and returns per-view 4x4
    rigid transformations that bring each view into a common canonical frame.
    """

    def __init__(self, model, cfg, rap_dir: str, device: str,
                 sampling_steps: int, rigidity_forcing: bool) -> None:
        self.model = model
        self.cfg = cfg
        self.rap_dir = rap_dir
        self.device = device
        self.sampling_steps = sampling_steps
        self.rigidity_forcing = rigidity_forcing
        self.last_chamfer_d = float("nan")

    def solve(
        self,
        pcds: List,
        voxel_m: float,
        max_points_per_part: int = 500,
    ) -> List[np.ndarray]:
        """Register multiple point clouds and return per-view 4x4 poses.

        Parameters
        ----------
        pcds : list[open3d.geometry.PointCloud]
            Input point clouds (one per view). Already in a common rough
            frame (e.g. floor-up).
        voxel_m : float
            Voxel size for downsampling before feeding to the model.
        max_points_per_part : int
            Max points per part after downsampling (RAP's default is 500).

        Returns
        -------
        list[np.ndarray]
            Per-view 4x4 transformation matrices (world-from-view).
        """
        import torch
        import open3d as o3d

        n_views = len(pcds)
        if n_views < 2:
            return [np.eye(4, dtype=np.float64)]

        parts_points = []
        for pcd in pcds:
            down = pcd.voxel_down_sample(voxel_m)
            pts = np.asarray(down.points, dtype=np.float32)
            if len(pts) > max_points_per_part:
                indices = np.random.choice(len(pts), max_points_per_part, replace=False)
                pts = pts[indices]
            parts_points.append(pts)

        points_per_part = [len(p) for p in parts_points]
        total_points = sum(points_per_part)

        all_pts = np.concatenate(parts_points, axis=0)
        pts_tensor = torch.from_numpy(all_pts).to(self.device)

        ppp = torch.zeros(1, n_views, dtype=torch.long, device=self.device)
        for i, c in enumerate(points_per_part):
            ppp[0, i] = c

        with torch.inference_mode():
            generated = self._generate(pts_tensor, ppp)

        T_list = self._recover_poses(parts_points, generated, ppp)
        return T_list

    def _generate(self, pts: "torch.Tensor", ppp: "torch.Tensor") -> "torch.Tensor":
        """Run the RAP flow-matching sampler."""
        import torch

        noise = torch.randn_like(pts)

        sampler = self.model
        if hasattr(sampler, 'sampler'):
            sampler = sampler.sampler

        n_steps = self.sampling_steps
        dt = 1.0 / n_steps
        x_t = noise.clone()

        batch = {
            "points": pts.unsqueeze(0) if pts.dim() == 2 else pts,
            "points_per_part": ppp,
        }

        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.tensor([t_val], device=pts.device, dtype=pts.dtype)

            with torch.no_grad():
                velocity = self.model.predict_velocity(x_t.unsqueeze(0), t_tensor, batch)
                if velocity.dim() == 3:
                    velocity = velocity.squeeze(0)

            x_t = x_t + velocity * dt

        if self.rigidity_forcing:
            try:
                from rectified_point_flow.procrustes import rigidify_prediction_with_procrustes
                x_t = rigidify_prediction_with_procrustes(
                    x_t, pts, ppp,
                )
            except Exception:
                LOG.warning("Rigidity forcing failed; using raw prediction")

        return x_t

    def _recover_poses(
        self,
        original_parts: List[np.ndarray],
        generated: "torch.Tensor",
        ppp: "torch.Tensor",
    ) -> List[np.ndarray]:
        """Recover per-view SE(3) via Procrustes between original and generated."""
        import torch

        points_per_part = ppp[0].cpu().tolist()
        gen_np = generated.cpu().float().numpy()

        T_list = []
        offset = 0
        for i, n_pts in enumerate(points_per_part):
            if n_pts == 0:
                T_list.append(np.eye(4, dtype=np.float64))
                continue
            src = original_parts[i].astype(np.float64)
            tgt = gen_np[offset:offset + n_pts].astype(np.float64)
            offset += n_pts

            src_c = src.mean(axis=0, keepdims=True)
            tgt_c = tgt.mean(axis=0, keepdims=True)
            src_centered = src - src_c
            tgt_centered = tgt - tgt_c

            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = tgt_c.squeeze() - R @ src_c.squeeze()

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            T_list.append(T)

        return T_list

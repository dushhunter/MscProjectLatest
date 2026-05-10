"""StoneVol_main-only dataset (not present in SPIdepth-main).

This file defines `StoneDataset`, a dataset class that:
- loads a *temporal triplet* of RGB frames (e.g., t-1, t, t+1)
- loads a *foreground mask* for the target frame t
- provides camera intrinsics K / inv_K at the required training scale(s)

It is designed to work with the `MonoDatasetMultiCam` base class, which reads a
per-folder intrinsics file (KV format).
"""

from __future__ import absolute_import, division, print_function  # py2/3 compatibility imports

import os  # filesystem path joins
import random  # random augmentation toggles
import numpy as np  # intrinsics matrix creation + pseudo-inverse
from PIL import Image  # PIL image ops (flip, interpolation enum)

import torch  # tensors returned to the training loop
from torchvision import transforms  # ToTensor + Resize transforms

from .mono_dataset_mc import MonoDatasetMultiCam, pil_loader  # base dataset + consistent image loader


class StoneDataset(MonoDatasetMultiCam):
    """Monocular stone dataset with per-frame masks.

    Expected layout under ``data_path``:
        stone_01/
            images/image_00.png
            images/image_01.png
            ...
            masks/mask_0001.png
            masks/mask_0002.png
            ...

    Split files (e.g., splits/stone/train_files.txt) should have lines of the form:
        stone_01 1
        where the second token is an integer *target* frame index.

        Neighboring frames are obtained by adding `frame_idxs` offsets (e.g., -1, 0, +1).
        Filenames are assumed to follow 4-digit frame indices for masks:
            - masks: ``mask_{frame_idx:04d}{mask_ext}``
        Legacy 2-digit masks are also supported as a fallback.
    """

    def __init__(self, *args, img_prefix="image", mask_prefix="mask", mask_ext=".png",
                 use_mask=False, use_gt_depth=False, gt_depth_path=None,
                 gt_depth_subdir="data_depth_annotated/train/groundtruth",
                 gt_depth_encoding="auto", gt_depth_scale=100000.0, **kwargs):
        # `*args` / `**kwargs` are forwarded to the base class. For StoneDataset, the
        # base constructor comes from MonoDatasetMultiCam and typically includes:
        #   (intrinsics_file_path, data_path, filenames, height, width, frame_idxs, num_scales, ...)
        super(StoneDataset, self).__init__(*args, **kwargs)

        self.img_prefix = img_prefix  # filename prefix for RGB images (legacy, not used for 4-digit naming)
        self.mask_prefix = mask_prefix  # filename prefix for masks (usually "mask")
        self.mask_ext = mask_ext  # file extension used for masks on disk (e.g., ".png")
        self.use_mask = use_mask  # whether to load per-frame masks
        self.use_gt_depth = use_gt_depth  # whether to load GT metric depth maps
        self.gt_depth_path = gt_depth_path  # root path containing data_depth_annotated/
        self.gt_depth_subdir = gt_depth_subdir  # relative GT root, supports custom folders
        self.gt_depth_encoding = gt_depth_encoding  # auto | uint16 | float32_rgba
        self.gt_depth_scale = float(gt_depth_scale)  # uint16 PNG → metres divisor (default 100000)
        self.interp_mask = Image.NEAREST  # keep masks discrete (no interpolation blending)
        self.mask_resize = {}  # per-scale resize transforms for masks

        for i in range(self.num_scales):  # build a resize op for each pyramid scale
            s = 2 ** i  # downscale factor at this pyramid level
            self.mask_resize[i] = transforms.Resize(  # torch/torchvision transform object
                (self.height // s, self.width // s), interpolation=self.interp_mask  # (H,W) at this scale
            )

    def preprocess(self, inputs, color_aug):
        """Convert PIL images to tensors and build image/mask pyramids.

        The training pipeline expects `inputs` to include:
        - color images at each requested `frame_id` (e.g., -1, 0, +1)
        - a mask for the target frame
        - K / inv_K for each scale
        This method handles the color/mask pyramids and converts them to tensors.
        """

        # Color pyramid + augmentation (mostly matches the parent dataset behavior)
        for k in list(inputs):  # iterate over a snapshot because we add new keys
            frame = inputs[k]  # value can be a PIL image or already-resized image
            if "color" in k:  # keys are like ("color", frame_id, scale)
                n, im, i = k  # unpack the tuple key
                for i in range(self.num_scales):  # create a pyramid across `num_scales`
                    # resize from previous scale (i-1). For i==0, i-1 == -1 is the original resolution.
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # Mask pyramid (NEAREST resize) - start from the native-resolution (-1) mask.
        # We only have a mask for the *target* frame (frame_id=0). We build a pyramid
        # so the loss can use masks at the same scales as the resized images.
        # Note: masks are stored as ("mask", 0, -1) initially, and then expanded to
        # ("mask", 0, 0), ("mask", 0, 1), ...
        for k in list(inputs):  # again, iterate on a snapshot as we add new keys
            frame = inputs[k]  # (unused directly here; kept for symmetry)
            if "mask" in k:  # keys are like ("mask", 0, -1)
                n, im, i = k  # unpack key (n="mask", im=0, i=-1)
                for scale in range(self.num_scales):  # create mask pyramid
                    # resize from previous scale (scale-1). For scale==0, scale-1 == -1 is original.
                    inputs[(n, im, scale)] = self.mask_resize[scale](inputs[(n, im, scale - 1)])

        for k in list(inputs):  # convert PIL images to tensors
            f = inputs[k]  # f is a PIL image for colors/masks, or numpy/torch for K etc.
            if "color" in k:  # handle RGB images
                n, im, i = k  # e.g., ("color", 0, 0)
                #inputs[(n, im, i)] = self.to_tensor(f)  # RGB -> float tensor in [0,1]
                #inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))  # augmented view used for training
                ###CURSOR >>>
                inputs[(n, im, i)] = self.to_tensor(f)  # RGB -> float tensor in [0,1]
                aug_tensor = self.to_tensor(color_aug(f))  # augmented view used for training
                if self.is_train:  # only keep augmented view during training to save memory
                    noise = torch.randn_like(aug_tensor) * 0.02  # small random noise for augmentation
                    aug_tensor = torch.clamp(aug_tensor + noise, 0.0, 1.0)  # add noise and clamp to [0,1]
                inputs[(n + "_aug", im, i)] = aug_tensor  # store augmented tensor
                ###<<<CURSOR
            if "mask" in k:  # handle mask images
                n, im, i = k  # e.g., ("mask", 0, 0)
                if i >= 0:  # only keep pyramid scales; do not tensorize the raw (-1) entry
                    mask_tensor = transforms.ToTensor()(f)  # grayscale PIL -> 1xHxW float tensor
                    inputs[(n, im, i)] = (mask_tensor > 0.5).float()  # binarize mask for stable losses

    def __getitem__(self, index):
        """Return one training sample (a dict of tensors).

        The returned dict contains:
        - ("color", frame_id, scale) and ("color_aug", frame_id, scale) tensors
        - ("mask", 0, scale) tensors for the target frame only
        - ("K", scale), ("K3x3", scale), ("inv_K", scale) tensors
        """

        inputs = {}  # will be populated with images, masks, intrinsics, and optional GT depth

        do_color_aug = self.is_train and random.random() > 0.5  # apply color jitter only during training
        do_flip = self.is_train and random.random() > 0.5  # random horizontal flip only during training

        line = self.filenames[index].split()  # split line like "stone_01 1" into [folder, frame]
        folder = line[0]  # sequence folder name (e.g., "stone_01")
        frame_index = int(line[1]) if len(line) > 1 else 0  # target frame index within the folder

        # Load target and source colors. We store them at scale=-1 first (native/original resolution).
        for i in self.frame_idxs:  # e.g., [-1, 0, +1]
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, do_flip)  # PIL RGB

        # Load target mask only when enabled (frame_id=0); source frames do not need masks.
        if self.use_mask:
            inputs[("mask", 0, -1)] = self.get_mask(folder, frame_index, do_flip)  # PIL grayscale

        # Intrinsics per scale (use per-folder key). K values are stored in normalized units
        # in the file, then scaled here by the current training resolution for each pyramid scale.
        for scale in range(self.num_scales):  # typically 1 scale in this repo (scale=0)
            K = self.get_intrinsics(folder).copy()  # 3x4 normalized K template (float32)
            K[0, :] *= self.width // (2 ** scale)  # scale fx and cx by image width at this scale
            K[1, :] *= self.height // (2 ** scale)  # scale fy and cy by image height at this scale

            inv_K = np.linalg.pinv(K)  # pseudo-inverse used for backprojection
            K3x3 = K[:3, :3].copy()  # 3x3 K sometimes used by other parts of the code

            inputs[("K", scale)] = torch.from_numpy(K)  # numpy -> torch tensor
            inputs[("K3x3", scale)] = torch.from_numpy(K3x3)  # numpy -> torch tensor
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)  # numpy -> torch tensor

        # Build a color augmentation function (jitter) if enabled for this sample.
        color_aug = (
            transforms.ColorJitter(  # torchvision jitter transform
                self.brightness, self.contrast, self.saturation, self.hue  # augmentation magnitudes from base class
            )
            if do_color_aug  # enable only for ~50% of training samples
            else (lambda x: x)  # identity function when augmentation is off
        )

        self.preprocess(inputs, color_aug)  # build pyramids + convert to tensors

        # Drop raw (-1) entries to save memory. After preprocess, we only keep scale>=0 tensors.
        for i in self.frame_idxs:  # for each frame id used in this sample
            del inputs[("color", i, -1)]  # remove native-resolution RGB PIL entry
            del inputs[("color_aug", i, -1)]  # remove native-resolution augmented PIL entry
        if self.use_mask and (("mask", 0, -1) in inputs):
            del inputs[("mask", 0, -1)]  # remove native-resolution mask PIL entry

        # Load GT metric depth for the target frame (after preprocess, so not part of image pyramid).
        # Stored as a float32 tensor [1, H, W] in metres; used by the trainer for supervised loss.
        if self.use_gt_depth and self.gt_depth_path is not None:
            depth_np = self.get_gt_depth(folder, frame_index, do_flip)  # float32 metres
            inputs["depth_gt"] = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        return inputs  # dict consumed by Trainer and the networks

    def get_color(self, folder, frame_index, do_flip):
        """Load one RGB frame as a PIL image.

        stone_syn_dataset naming: images are stored directly in the folder root
        as 4-digit zero-padded files, e.g. stone_01/0001.png.
        """
        fname = f"{frame_index:04d}.png"  # e.g. 0001.png, 0042.png
        image_path = os.path.join(self.data_path, folder, fname)
        color = pil_loader(image_path)  # load RGB image via shared loader
        if do_flip:  # optionally mirror image horizontally
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color  # PIL.Image in RGB mode

    def _decode_float32_rgba_depth(self, depth_img):
        """Decode lossless float32 depth packed into RGBA PNG channels."""
        rgba = np.array(depth_img.convert("RGBA"), dtype=np.uint8)
        h, w, c = rgba.shape
        if c != 4:
            raise ValueError("Expected RGBA depth image with 4 channels")
        # Depth was encoded as little-endian float32 bytes [R, G, B, A].
        depth = rgba.reshape(-1, 4).view("<f4").reshape(h, w).copy()
        return depth

    def get_gt_depth(self, folder, frame_index, do_flip):
        """Load GT metric depth PNG for the target frame.

        GT depths live at: {gt_depth_path}/{gt_depth_subdir}/{folder}/depth_{frame:04d}.png
        Supports:
          - uint16 PNG (metres = value / gt_depth_scale)
          - float32 RGBA PNG (lossless packed float32 metres)
        """
        fname = f"depth_{frame_index:04d}.png"  # e.g. depth_0001.png
        depth_path = os.path.join(
            self.gt_depth_path,
            self.gt_depth_subdir,
            folder, fname,
        )
        # Load raw image — do NOT use pil_loader which converts to RGB.
        with open(depth_path, "rb") as f:
            depth_pil = Image.open(f)
            depth_pil = depth_pil.copy()  # detach from file handle
        if do_flip:
            depth_pil = depth_pil.transpose(Image.FLIP_LEFT_RIGHT)

        encoding = self.gt_depth_encoding
        if encoding == "float32_rgba":
            return self._decode_float32_rgba_depth(depth_pil).astype(np.float32)

        if encoding == "uint16":
            return np.array(depth_pil, dtype=np.float32) / self.gt_depth_scale

        # Auto-detect: RGBA means float32-packed depth; otherwise legacy uint16/int depth.
        if depth_pil.mode == "RGBA":
            return self._decode_float32_rgba_depth(depth_pil).astype(np.float32)

        return np.array(depth_pil, dtype=np.float32) / self.gt_depth_scale

    def get_mask(self, folder, frame_index, do_flip):
        """Load one target-frame mask as a PIL image (grayscale)."""
        mask_dir = os.path.join(self.data_path, folder, "masks")
        # Prefer 4-digit names (mask_0001.png), then fall back to legacy 2-digit names.
        candidate_names = [
            self._format_frame_name(frame_index, prefix=self.mask_prefix, ext=self.mask_ext, pad=4),
            self._format_frame_name(frame_index, prefix=self.mask_prefix, ext=self.mask_ext, pad=2),
        ]
        mask_path = None
        for fname in candidate_names:
            candidate_path = os.path.join(mask_dir, fname)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        if mask_path is None:
            raise FileNotFoundError(
                "Mask not found for frame {} in {}. Tried: {}".format(
                    frame_index,
                    mask_dir,
                    ", ".join(candidate_names),
                )
            )
        mask = pil_loader(mask_path).convert("L")  # ensure 1-channel grayscale
        if do_flip:  # apply same flip as the RGB image so they stay aligned
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return mask  # PIL.Image in L mode

    def get_intrinsics(self, folder):
        """Return the normalized 3x4 intrinsics matrix for this folder.

        `MonoDatasetMultiCam` populates `self.KV_intrinsics_dict` from the intrinsics file.
        We key by folder name so different sequences *can* have different intrinsics.
        """
        return self.KV_intrinsics_dict[folder]  # numpy float32 array of shape (3,4)

    def get_intrinsics_map(self, file_name):
        """Parse a KV intrinsics file into a dict.

        File format (one line per folder):
            <folder> <fx_norm> <fy_norm> <cx_norm> <cy_norm>

        Values are *normalized* by image width/height, so they are resolution-independent.
        We convert them into a 3x4 matrix format used throughout the project.
        """
        lines = self._read_lines(file_name)  # read all lines from file
        KV_intrinsics_dict = {}  # folder -> K (3x4)
        for line in lines:  # iterate each line
            parts = line.strip().split()  # split by whitespace
            if len(parts) < 5:  # skip malformed/empty/comment lines
                continue
            folder_key = parts[0]  # e.g., "stone_01"
            fx = float(parts[1])  # normalized focal length x: fx / W
            fy = float(parts[2])  # normalized focal length y: fy / H
            cx = float(parts[3])  # normalized principal point x: cx / W
            cy = float(parts[4])  # normalized principal point y: cy / H
            KV_intrinsics_dict[folder_key] = np.array(  # create K in a 3x4 "projection" style matrix
                [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float32
            )
        return KV_intrinsics_dict  # mapping used by get_intrinsics()

    @staticmethod
    def _read_lines(file_name):
        """Read a text file and return its lines."""
        with open(file_name, "r") as f:  # open in text read mode
            return f.readlines()  # return list[str]

    @staticmethod
    def _format_frame_name(frame_index, prefix="image", ext=".png", pad=4):
        """Format filenames like image_0001.png or mask_0009.png."""
        return f"{prefix}_{frame_index:0{pad}d}{ext}"  # configurable zero-padded frame index

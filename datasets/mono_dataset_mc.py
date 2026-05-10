"""Multi-camera (KV-intrinsics) dataset base class.

This file defines `MonoDatasetMultiCam`, which is similar in spirit to the common
`MonoDataset` used in Monodepth2-style code, but differs in one key aspect:

- Intrinsics are *not hardcoded* in code.
- Instead, they are loaded from a KV-style intrinsics text file, typically one
    line per sequence/folder:

        <folder_key> <fx_norm> <fy_norm> <cx_norm> <cy_norm>

where the values are normalized by original image width/height (so they are
resolution-independent). The dataset then scales them to the current training
resolution (and to each image pyramid scale).

`StoneDataset` and `MCDataset` inherit from this base class.
"""

from __future__ import absolute_import, division, print_function  # python2/3 compatibility

import os  # filesystem utilities (kept for subclasses)
import random  # data augmentation randomness
import numpy as np  # matrix math for intrinsics and pseudo-inverse
import copy  # deep/shallow copy helpers (kept for compatibility)
from PIL import Image  # PIL.Image utilities and interpolation enums

import torch  # tensors returned to the training loop
import torch.utils.data as data  # PyTorch dataset base class
from torchvision import transforms  # image transforms: Resize/ToTensor/ColorJitter


def pil_loader(path):
    """Load an image path into a PIL RGB image.

    Using `open(path, 'rb')` avoids PIL resource warnings by ensuring the file
    handle is managed correctly.
    """

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:  # open file in binary mode
        with Image.open(f) as img:  # let PIL parse the image
            return img.convert('RGB')  # always return 3-channel RGB


class MonoDatasetMultiCam(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 KV_intrinsics_file,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDatasetMultiCam, self).__init__()  # initialize torch Dataset

        # Parse the KV intrinsics file into a dict mapping folder_key -> normalized K (3x4).
        # NOTE: `get_intrinsics_map` is intentionally left for subclasses to implement so
        # they can define the exact parsing logic and expected key format.
        self.KV_intrinsics_dict = self.get_intrinsics_map(KV_intrinsics_file)

        self.data_path = data_path  # root directory of the dataset
        self.filenames = filenames  # list[str] of split lines (folder + frame index)
        self.height = height  # training input height (after resizing)
        self.width = width  # training input width (after resizing)
        self.num_scales = num_scales  # number of image pyramid scales to generate
        # Pillow>=10 removed Image.ANTIALIAS; use LANCZOS (high-quality downsampling)
        # with a backwards-compatible fallback.
        self.interp = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else getattr(Image, "LANCZOS", Image.BICUBIC)
        )

        self.frame_idxs = frame_idxs  # e.g., [0, -1, 1]

        self.is_train = is_train  # toggles augmentation logic
        self.img_ext = img_ext  # image extension expected on disk (e.g., .png)

        self.loader = pil_loader  # function used to load RGB images
        self.to_tensor = transforms.ToTensor()  # converts PIL -> float tensor in [0,1]

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)  # multiplicative brightness range
            self.contrast = (0.8, 1.2)  # multiplicative contrast range
            self.saturation = (0.8, 1.2)  # multiplicative saturation range
            self.hue = (-0.1, 0.1)  # additive hue shift range
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)  # validate API accepts tuples
        except TypeError:
            # Older torchvision versions expect scalar magnitudes instead of ranges.
            self.brightness = 0.2  # brightness jitter magnitude
            self.contrast = 0.2  # contrast jitter magnitude
            self.saturation = 0.2  # saturation jitter magnitude
            self.hue = 0.1  # hue jitter magnitude

        self.resize = {}  # maps scale index -> Resize transform
        for i in range(self.num_scales):  # create a resize op for each pyramid scale
            s = 2 ** i  # downsample factor at this scale
            self.resize[i] = transforms.Resize(  # torchvision resize transform
                (self.height // s, self.width // s),  # output (H, W) for this scale
                interpolation=self.interp  # interpolation method for RGB images
            )

        # self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # Build the color pyramid first (PIL domain).
        for k in list(inputs):  # use list() because we add new keys during iteration
            frame = inputs[k]  # the stored value, usually a PIL image
            if "color" in k:  # keys like ("color", frame_id, scale)
                n, im, i = k  # unpack: n="color", im=frame_id, i=scale
                for i in range(self.num_scales):  # produce scale 0..num_scales-1
                    # resize from previous scale (i-1); i==0 uses original (-1)
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # Convert PIL images to tensors and generate augmented copies.
        for k in list(inputs):  # second pass: tensorization
            f = inputs[k]  # PIL image
            if "color" in k:
                n, im, i = k  # unpack key
                inputs[(n, im, i)] = self.to_tensor(f)  # RGB tensor
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))  # augmented RGB tensor

    def __len__(self):
        return len(self.filenames)  # number of split entries

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}  # dict of sample contents consumed by the Trainer

        do_color_aug = self.is_train and random.random() > 0.5  # 50% of training samples get color jitter
        do_flip = self.is_train and random.random() > 0.5  # 50% of training samples get horizontal flip

        line = self.filenames[index].split()  # parse split line (folder + frame index)
        folder = line[0]  # dataset sequence folder or key

        if len(line) == 2:  # most splits have: "folder frame_index"
            frame_index = int(line[1])  # target frame index
        else:
            frame_index = 0  # fallback default when frame index is missing

        # if len(line) == 3:
        #     side = line[2]
        # else:
        #     side = None

        for i in self.frame_idxs:  # iterate requested temporal offsets, e.g., [0, -1, +1]
            # frame_idxs: [0,-1,1]
            # if i == "s":
            #     other_side = {"r": "l", "l": "r"}[side]
            #     inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            # else:
                # inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, do_flip)  # PIL RGB at native res

        # Adjust intrinsics to match each scale in the pyramid.
        # The intrinsics file stores normalized values, so here we multiply by the
        # actual training resolution for each scale.
        for scale in range(self.num_scales):
            # K = self.K.copy()

            # Derive the key used in the KV intrinsics dict.
            # If `folder` is already a simple key (e.g., "stone_01"), this is a no-op.
            # If `folder` contains path separators, this extracts the last component.
            intrinsics_key = folder[folder.rfind('/')+1:]

            # Fetch the normalized 3x4 K template and copy it to avoid mutating the cached dict value.
            K = self.get_intrinsics(intrinsics_key).copy()  # copy prevents in-place scaling side effects

            # Scale first row (fx, cx) by width and second row (fy, cy) by height.
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)  # pseudo-inverse used by backprojection
            K3x3 = K[:3, :3].copy()  # convenient 3x3 copy used by some modules

            inputs[("K", scale)] = torch.from_numpy(K)  # numpy -> torch
            inputs[("K3x3", scale)] = torch.from_numpy(K3x3)  # numpy -> torch
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)  # numpy -> torch

        if do_color_aug:  # build a per-sample augmentation function
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)  # identity when augmentation is disabled

        self.preprocess(inputs, color_aug)  # build pyramid + convert to tensors

        for i in self.frame_idxs:  # remove native-resolution entries to save memory
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, side, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # if "s" in self.frame_idxs:
        #     stereo_T = np.eye(4, dtype=np.float32)
        #     baseline_sign = -1 if do_flip else 1
        #     side_sign = -1 if side == "l" else 1
        #     stereo_T[0, 3] = side_sign * baseline_sign * 0.1

        #     inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs  # returned dict is consumed by the Trainer

    def get_color(self, folder, frame_index, do_flip):
        raise NotImplementedError  # implemented in child datasets (StoneDataset, MCDataset)

    # def check_depth(self):
    #     raise NotImplementedError

    def get_depth(self, folder, frame_index, do_flip):
        raise NotImplementedError  # optional; used only if GT depth is available

    def get_intrinsics_map(self, file_name):
        raise NotImplementedError  # implemented in child datasets to parse KV intrinsics file



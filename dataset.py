"""
Dataset for ProObjaverse-300K in sharded tar format.

Data layout:
  renders_random_env/
    shard-XXXX/
      {uid}.tar   -> {uid}/{idx:03d}.json  (camera meta)
                     {uid}/{idx:03d}.rgba.webp (RGBA image, 1024×1024)
  lh-slats/
    shard-XXXX/
      {uid}.npz   -> feats: (N, 8) float32
                     coords: (N, 3) uint8  (voxel coords in [0,63])
"""

import os
import io
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import webdataset as wds


# ---------------------------------------------------------------------------
# Helper: read all views from a uid.tar via webdataset.
# webdataset groups files by key = path-before-first-dot-in-basename, so
#   uid/000.json + uid/000.rgba.webp  →  one sample with keys "json" and "rgba.webp"
# ---------------------------------------------------------------------------

def _load_views_from_tar(tar_path: str) -> list:
    """
    Stream a uid.tar with webdataset and return a list of raw view dicts.
    Each dict has keys: 'json' (bytes), 'rgba.webp' (bytes).
    Incomplete samples (missing either file) are silently skipped.
    """
    views = []
    for sample in wds.WebDataset(
        tar_path,
        shardshuffle=False,
        nodesplitter=lambda src, **kw: src,
        workersplitter=lambda src: src,
        empty_check=False,
    ):
        if "json" in sample and "rgba.webp" in sample:
            views.append({"json": sample["json"], "rgba.webp": sample["rgba.webp"]})
    return views


# ---------------------------------------------------------------------------
# Helper: prepare_batch_images (crop foreground bbox, pad to square, resize)
# ---------------------------------------------------------------------------

def prepare_batch_images(
    imgs: torch.Tensor,   # (B, 3, H, W)  float32 in [0,1]
    masks: torch.Tensor,  # (B, 1, H, W)  float32 in [0,1]
    resolution: int = 518,
    no_background: bool = True,
    padding_factor: float = 1.1,
) -> torch.Tensor:
    """
    Vectorised crop + square-pad + resize.
    Returns (B, 3, resolution, resolution).
    """
    B, C, H, W = imgs.shape
    assert C == 3 and masks.shape == (B, 1, H, W)

    if no_background:
        imgs = imgs * masks

    inf = torch.tensor(1e5, device=imgs.device, dtype=torch.long)
    mask_bool = (masks[:, 0] > 0.5)  # (B, H, W)

    ys = torch.arange(H, device=imgs.device).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=imgs.device).view(1, 1, W).expand(B, H, W)

    y0 = torch.where(mask_bool, ys, inf).flatten(1).min(1)[0].clamp(max=H - 1).float()
    x0 = torch.where(mask_bool, xs, inf).flatten(1).min(1)[0].clamp(max=W - 1).float()
    y1 = torch.where(mask_bool, ys, torch.zeros_like(ys) - inf).flatten(1).max(1)[0].clamp(min=0).float()
    x1 = torch.where(mask_bool, xs, torch.zeros_like(xs) - inf).flatten(1).max(1)[0].clamp(min=0).float()

    no_fg = (mask_bool.sum(dim=[1, 2]) == 0)
    y0 = torch.where(no_fg, torch.zeros_like(y0), y0)
    x0 = torch.where(no_fg, torch.zeros_like(x0), x0)
    y1 = torch.where(no_fg, torch.full_like(y1, H - 1), y1)
    x1 = torch.where(no_fg, torch.full_like(x1, W - 1), x1)

    cy = (y0 + y1) * 0.5
    cx = (x0 + x1) * 0.5
    side = torch.max(y1 - y0, x1 - x0) * padding_factor

    y0n = (cy - side / 2).clamp(0, H - 1)
    y1n = (cy + side / 2).clamp(0, H - 1)
    x0n = (cx - side / 2).clamp(0, W - 1)
    x1n = (cx + side / 2).clamp(0, W - 1)

    scale_y = (y1n - y0n) / (H - 1)
    scale_x = (x1n - x0n) / (W - 1)
    trans_y = (y0n + y1n - (H - 1)) / (H - 1)
    trans_x = (x0n + x1n - (W - 1)) / (W - 1)

    A = torch.zeros(B, 2, 3, device=imgs.device, dtype=imgs.dtype)
    A[:, 0, 0] = scale_x
    A[:, 1, 1] = scale_y
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y

    grid = F.affine_grid(A, [B, C, resolution, resolution], align_corners=False)
    out = F.grid_sample(imgs, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TarDataset(Dataset):
    """
    Dataset that reads RGBA images from sharded tar files and slat NPZs.

    Args:
        data_root: Path to ProObjaverse-300K directory.
        num_views:  Number of views to sample per item.
        random_sample: If > 0, randomly sub-sample sparse-latent points.
        image_size: Resize images to this resolution before returning
                    (applied after cropping; 0 = keep original 1024).
    """

    def __init__(
        self,
        data_root: str,
        num_views: int = 6,
        random_sample: int = -1,
        image_size: int = 518,
    ):
        self.data_root = data_root
        self.num_views = num_views
        self.random_sample = random_sample
        self.image_size = image_size

        self.samples = self._build_index()
        print(f"[TarDataset] {len(self.samples)} samples found.")

    # ------------------------------------------------------------------
    def _build_index(self):
        renders_dir = os.path.join(self.data_root, "renders_random_env")
        slats_dir = os.path.join(self.data_root, "lh-slats")

        # uid -> slat_path
        slat_map: dict[str, str] = {}
        for shard in os.listdir(slats_dir):
            shard_path = os.path.join(slats_dir, shard)
            if not os.path.isdir(shard_path):
                continue
            for fname in os.listdir(shard_path):
                if fname.endswith(".npz"):
                    uid = fname[:-4]
                    slat_map[uid] = os.path.join(shard_path, fname)

        samples = []
        for shard in os.listdir(renders_dir):
            shard_path = os.path.join(renders_dir, shard)
            if not os.path.isdir(shard_path):
                continue
            for fname in os.listdir(shard_path):
                if not fname.endswith(".tar"):
                    continue
                uid = fname[:-4]
                if uid not in slat_map:
                    continue
                samples.append({
                    "uid": uid,
                    "tar_path": os.path.join(shard_path, fname),
                    "slat_path": slat_map[uid],
                })

        return samples

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                print(f"[TarDataset] Error loading item {idx}: {e}")
                idx = random.randint(0, len(self.samples) - 1)

    def _load_item(self, idx):
        sample = self.samples[idx]
        uid = sample["uid"]

        # ---- load slat ------------------------------------------------
        slat_data = np.load(sample["slat_path"])
        target_feats = torch.from_numpy(slat_data["feats"].copy())   # (N, 8)
        target_coords = torch.from_numpy(
            slat_data["coords"].astype(np.int32)
        )  # (N, 3)

        # Add leading batch-index placeholder (0) → (N, 4); collate will fill it
        target_coords = torch.cat(
            [torch.zeros_like(target_coords[:, :1]), target_coords], dim=-1
        )

        if self.random_sample > 0:
            n_pts = min(self.random_sample, target_feats.shape[0])
            idxs = np.random.choice(target_feats.shape[0], n_pts, replace=False)
            target_feats = target_feats[idxs]
            target_coords = target_coords[idxs]

        if torch.isnan(target_feats).any():
            raise ValueError("NaN in target_feats")

        # ---- load tar via webdataset ------------------------------------
        all_views = _load_views_from_tar(sample["tar_path"])
        if not all_views:
            raise ValueError(f"No views in tar: {sample['tar_path']}")

        selected = random.sample(all_views, self.num_views)

        images, extrinsics, intrinsics = [], [], []
        for view in selected:
            meta = json.loads(view["json"])
            img = Image.open(io.BytesIO(view["rgba.webp"])).convert("RGBA")
            img_tensor = TF.to_tensor(img)  # (4, H, W)  in [0,1]

            images.append(img_tensor)
            extrinsics.append(
                torch.tensor(meta["extrinsic"], dtype=torch.float32)
            )  # (4, 4)
            intrinsics.append(
                torch.tensor(meta["intrinsic"], dtype=torch.float32)
            )  # (3, 3)

        images = torch.stack(images)        # (N, 4, H, W)
        extrinsics = torch.stack(extrinsics)  # (N, 4, 4)
        intrinsics = torch.stack(intrinsics)  # (N, 3, 3)

        alpha = images[:, 3:]                 # (N, 1, H, W)
        ref_image = images[:, :3] * alpha     # (N, 3, H, W) black-bg composite

        if self.image_size > 0:
            h, w = ref_image.shape[-2:]
            if h != self.image_size or w != self.image_size:
                ref_image = F.interpolate(
                    ref_image, self.image_size, mode="bilinear", align_corners=False
                )
                alpha = F.interpolate(alpha, self.image_size, mode="nearest")

        return {
            "ref_image": ref_image,          # (N, 3, H, W)
            "alpha": alpha,                  # (N, 1, H, W)
            "target_feats": target_feats,    # (N_pts, 8)
            "target_coords": target_coords,  # (N_pts, 4)  first col = 0
            "batch_extrinsics": extrinsics,  # (N, 4, 4)
            "batch_intrinsics": intrinsics,  # (N, 3, 3)
        }


# ---------------------------------------------------------------------------
# Collate function (handles sparse coords with proper batch indices)
# ---------------------------------------------------------------------------

def custom_collate(batch):
    """
    Collate a list of dataset items.
    Sparse coords get their batch dimension filled in here.
    """
    batched_feats, batched_coords = [], []
    batched_ref, batched_alpha = [], []
    batched_extrinsics, batched_intrinsics = [], []

    for b_idx, sample in enumerate(batch):
        # sparse latent
        feats = sample["target_feats"]
        coords = sample["target_coords"][..., 1:]  # drop placeholder → (N, 3)
        batch_col = torch.full((coords.shape[0], 1), b_idx, dtype=coords.dtype)
        batched_feats.append(feats)
        batched_coords.append(torch.cat([batch_col, coords], dim=1))  # (N, 4)

        batched_ref.append(sample["ref_image"])
        batched_alpha.append(sample["alpha"])
        batched_extrinsics.append(sample["batch_extrinsics"])
        batched_intrinsics.append(sample["batch_intrinsics"])

    return {
        "ref_image": torch.stack(batched_ref),            # (B, N, 3, H, W)
        "alpha": torch.stack(batched_alpha),              # (B, N, 1, H, W)
        "target_feats": torch.cat(batched_feats, dim=0),  # (sum_N, 8)
        "target_coords": torch.cat(batched_coords, dim=0),# (sum_N, 4)
        "batch_extrinsics": torch.stack(batched_extrinsics),  # (B, N, 4, 4)
        "batch_intrinsics": torch.stack(batched_intrinsics),  # (B, N, 3, 3)
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    data_root = sys.argv[1] if len(sys.argv) > 1 else "/root/public-read/ProObjaverse-300K"
    ds = TarDataset(data_root, num_views=6, random_sample=-1)
    sample = ds[0]
    print("Keys:", list(sample.keys()))
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    loader = DataLoader(ds, batch_size=2, collate_fn=custom_collate, num_workers=0)
    batch = next(iter(loader))
    print("\nBatch:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} {v.dtype}")

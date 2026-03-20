"""
Training script for the Sparse-Structure (SS) flow model.

Corresponds to the old train_vggt_lora_ss.py, adapted for:
  - TrellisVGGTTo3DPipeline  (sparse_structure_flow_model + sparse_structure_vggt_cond)
  - New dataset: sharded tar files in ProObjaverse-300K
  - VGGT loaded from "Stable-X/vggt-object-v0-1" (HuggingFace)

Trainable components:
  1. sparse_structure_vggt_cond  (ModulatedMultiViewCond, freshly initialised)
  2. LoRA adapters on sparse_structure_flow_model  (r=64, α=128)

VGGT and all other models are frozen.

Usage:
  torchrun --nproc_per_node=8 train_ss.py \
      --data_root /root/public-read/ProObjaverse-300K \
      --weights   microsoft/TRELLIS-image-large \
      --save_dir  checkpoints/ss-vggt-lora \
      [--resume   checkpoints/ss-vggt-lora/epoch=0-step=1000.ckpt]
"""

import os
os.environ["SPCONV_ALGO"] = 'native'
import sys
import argparse
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler

# ---- path setup -----------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wheels", "vggt"))

from dataset import TarDataset, custom_collate, prepare_batch_images
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.models.sparse_structure_flow import ModulatedMultiViewCond
from trellis import models as trellis_models
from torchvision import transforms

# VGGT
from wheels.vggt.vggt.models.vggt import VGGT


# ---------------------------------------------------------------------------
# LightningModule wrapper for SS training
# ---------------------------------------------------------------------------

class SSTrainer(pl.LightningModule):
    """
    Wraps the SS-flow-model + sparse_structure_vggt_cond for flow-matching training.
    """

    # DINOv2 normalisation (same as TrellisImageTo3DPipeline)
    _dino_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        ss_flow_model: nn.Module,
        ss_cond: nn.Module,
        image_cond_model: nn.Module,
        vggt_model: nn.Module,
        ss_encoder: nn.Module,
        ss_sampler,
        lr: float = 1e-4,
        cfg_drop_prob: float = 0.1,
        vggt_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.image_cond_model = image_cond_model
        self.vggt_model = vggt_model
        self.ss_encoder = ss_encoder
        self.ss_sampler = ss_sampler
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.vggt_dtype = vggt_dtype

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 encoding.  image: (B*N, 3, 518, 518) in [0,1], float32.
        Returns (B*N, n_tokens, 1024).
        """
        image = image.to(self.device)
        image = self._dino_transform(image)
        feats = self.image_cond_model(image, is_training=True)["x_prenorm"]
        feats = F.layer_norm(feats, feats.shape[-1:])
        return feats  # (B*N, n_tok, 1024)

    def _prepare_images(self, images: torch.Tensor, alpha: torch.Tensor):
        """
        Crop foreground, resize to 518.
        images: (B*N, 3, H, W); alpha: (B*N, 1, H, W)
        Returns (B*N, 3, 518, 518).
        """
        mask = alpha > 0.5
        out = prepare_batch_images(images, mask.float(), resolution=518, padding_factor=1.1)
        return out

    # ------------------------------------------------------------------
    def get_input(self, batch):
        """
        Prepare (targets, cond, noise) for the SS flow-matching loss.

        targets : (B, C, D, H, W) dense SS latents
        cond    : (B, C_cond) conditioning vector
        noise   : (B, C, D, H, W) Gaussian noise
        """
        images = batch["ref_image"].to(self.vggt_dtype)   # (B, N, 3, H, W)
        alpha  = batch["alpha"].to(torch.float32)          # (B, N, 1, H, W)

        b, n, c, h, w = images.shape
        # Randomly drop some views (1 … n)
        n_use = random.randint(1, n)
        images = images[:, :n_use]
        alpha  = alpha[:, :n_use]

        # Resize / prepare
        images_flat = F.interpolate(
            images.reshape(b * n_use, c, h, w), 518, mode="bilinear", align_corners=False
        )
        alpha_flat = F.interpolate(
            alpha.reshape(b * n_use, 1, h, w), 518, mode="nearest"
        )

        images_flat = self._prepare_images(
            images_flat.float(), alpha_flat.float()
        ).to(self.vggt_dtype)

        images_for_vggt = images_flat.reshape(b, n_use, c, 518, 518)

        # ---- VGGT features (frozen) -----------------------------------
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.vggt_dtype):
                aggregated_tokens_list, _ = self.vggt_model.aggregator(images_for_vggt)

        # ---- DINOv2 features (frozen) ---------------------------------
        with torch.no_grad():
            image_cond = self._encode_image(
                images_flat.float()
            ).reshape(b, n_use, -1, 1024)  # (B, N, n_tok, 1024)
            # Drop first 5 cls/reg tokens (same as old pipeline)
            image_cond = image_cond[:, :, 5:]

        # ---- Conditioning --------------------------------------------
        cond = self.ss_cond(aggregated_tokens_list, image_cond)
        if random.random() < self.cfg_drop_prob:
            cond = torch.zeros_like(cond)

        # ---- SS latent targets (from voxel grid) ---------------------
        target_coords = batch["target_coords"].to(self.device)  # (sum_N, 4)
        with torch.no_grad():
            ss = torch.zeros(
                b, 64, 64, 64, dtype=torch.long, device=self.device
            )
            ss.index_put_(
                (
                    target_coords[:, 0],
                    target_coords[:, 1],
                    target_coords[:, 2],
                    target_coords[:, 3],
                ),
                torch.tensor(1, dtype=ss.dtype, device=self.device),
            )
            ss = ss.unsqueeze(1).float()  # (B, 1, 64, 64, 64)
            targets = self.ss_encoder(
                ss.to(next(self.ss_encoder.parameters()).dtype),
                sample_posterior=False,
            )
            targets = targets.to(torch.float32)

        noise = torch.randn_like(targets)
        return targets, cond, noise

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        t = torch.rand(1).item()
        targets, cond, noise = self.get_input(batch)

        # Flow matching: x_t, ground-truth velocity
        x_t, gt_v = self.ss_sampler._get_model_gt(targets, t, noise)

        t_tensor = torch.tensor(
            [1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)

        loss = F.mse_loss(pred_v, gt_v, reduction="none")
        loss = loss[~torch.isnan(loss)].mean()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            k: v for k, v in state_dict.items()
            if k.startswith("ss_flow_model.") or k.startswith("ss_cond.")
        }

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lora_params = [p for p in self.ss_flow_model.parameters() if p.requires_grad]
        params = list(self.ss_cond.parameters()) + lora_params
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)


# ---------------------------------------------------------------------------
# Build pipeline / models
# ---------------------------------------------------------------------------

def build_models(weights_path: str, resume_path: str = None, local_rank: int = 0):
    """
    Load all required models and return an SSTrainer.
    weights_path: HuggingFace repo or local dir for microsoft/TRELLIS-image-large.
    """
    device = f"cuda:{local_rank}"

    # ---- Base TRELLIS pipeline (loads DINOv2, samplers, flow models) ----
    print(f"[rank {local_rank}] Loading TrellisImageTo3DPipeline from {weights_path} ...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(weights_path)
    pipeline.to(device)

    ss_flow_model = pipeline.models["sparse_structure_flow_model"]
    image_cond_model = pipeline.models["image_cond_model"]
    ss_sampler = pipeline.sparse_structure_sampler

    # ---- SS encoder (VAE encoder for voxel → latent) -------------------
    ss_encoder = trellis_models.from_pretrained(
        f"{weights_path}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(weights_path)
        else f"{weights_path}/ckpts/ss_enc_conv3d_16l8_fp16"
    )
    ss_encoder = ss_encoder.to(device).eval()
    for p in ss_encoder.parameters():
        p.requires_grad = False

    # ---- VGGT -----------------------------------------------------------
    print(f"[rank {local_rank}] Loading VGGT from Stable-X/vggt-object-v0-1 ...")
    vggt_model = VGGT.from_pretrained("Stable-X/vggt-object-v0-1")
    # Remove heads not needed for training
    del vggt_model.depth_head
    del vggt_model.track_head
    del vggt_model.point_head
    del vggt_model.camera_head
    vggt_model = vggt_model.to(device).eval()
    for p in vggt_model.parameters():
        p.requires_grad = False

    vggt_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(local_rank)[0] >= 8
        else torch.float16
    )

    # ---- Freeze SS flow model, then apply LoRA -------------------------
    ss_flow_model = ss_flow_model.to(device).eval()
    # ss_flow_model.convert_to_fp32()
    for p in ss_flow_model.parameters():
        p.requires_grad = False

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
    )
    ss_flow_model = get_peft_model(ss_flow_model, lora_cfg)
    ss_flow_model.print_trainable_parameters()

    # ---- DINOv2 frozen --------------------------------------------------
    image_cond_model = image_cond_model.to(device).eval()
    for p in image_cond_model.parameters():
        p.requires_grad = False

    # ---- sparse_structure_vggt_cond (freshly initialised) --------------
    ss_cond = ModulatedMultiViewCond(
        channels=1024,
        ctx_channels=3072,
        num_heads=16,
        mlp_ratio=4.0,
        attn_mode="full",
        use_checkpoint=False,
        use_rope=False,
        share_mod=False,
        qk_rms_norm=True,
        qk_rms_norm_cross=False,
    ).to(device).train()
    for p in ss_cond.parameters():
        p.requires_grad = True

    # ---- Optionally load checkpoint ------------------------------------
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"[rank {local_rank}] Resuming from {resume_path}")
        states = torch.load(resume_path, map_location="cpu")
        if "state_dict" in states:
            states = states["state_dict"]
        ss_flow_model.load_state_dict(
            {k.replace("ss_flow_model.", ""): v for k, v in states.items()},
            strict=False,
        )
        ss_cond.load_state_dict(
            {k.replace("ss_cond.", ""): v for k, v in states.items()},
            strict=False,
        )

    trainer_module = SSTrainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        image_cond_model=image_cond_model,
        vggt_model=vggt_model,
        ss_encoder=ss_encoder,
        ss_sampler=ss_sampler,
        vggt_dtype=vggt_dtype,
    )
    return trainer_module


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/public-read/ProObjaverse-300K")
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--save_dir", default="checkpoints/ss-vggt-lora")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accum_batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # ---- Distributed setup ---------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=3600),
        )

    # ---- Dataset -------------------------------------------------------
    dataset = TarDataset(
        data_root=args.data_root,
        num_views=args.num_views,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=custom_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Models --------------------------------------------------------
    module = build_models(
        weights_path=args.weights,
        resume_path=args.resume,
        local_rank=local_rank,
    )

    # ---- Logger --------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=args.save_dir.split("/")[-1],
        version="",
    )

    # ---- Callbacks -----------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        every_n_epochs=1,
        save_top_k=-1,
        save_weights_only=True,
    )
    swa_cb = StochasticWeightAveraging(swa_lrs=1e-2)

    # ---- Trainer -------------------------------------------------------
    trainer = pl.Trainer(
        devices=world_size,
        accelerator="cuda",
        max_epochs=args.max_epochs,
        precision=16,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb, swa_cb],
        accumulate_grad_batches=args.accum_batches,
        gradient_clip_val=0.5,
    )

    trainer.fit(module, dataloader)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

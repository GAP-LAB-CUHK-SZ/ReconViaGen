"""
Hybrid pipeline: ReconViaGen VGGT-based SS stage + TRELLIS.2 shape_slat / tex_slat stages.

Stage 1  – Sparse Structure (SS)  : TrellisVGGTTo3DPipeline  (ReconViaGen, VGGT-conditioned)
Stage 2  – Shape SLat             : Trellis2ImageTo3DPipeline (TRELLIS.2, DINOv3-conditioned)
Stage 3  – Texture SLat           : Trellis2ImageTo3DPipeline (TRELLIS.2, DINOv3-conditioned)
Stage 4  – Decode / GLB export    : Trellis2ImageTo3DPipeline (TRELLIS.2, o_voxel)
"""

import sys
import os

# ── Make trellis2 importable from wheels/TRELLIS.2 ───────────────────────────
_TRELLIS2_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'wheels', 'TRELLIS.2'))

if _TRELLIS2_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS2_ROOT)
# o_voxel is installed into the conda env (no extra sys.path needed)

# ── Standard imports ──────────────────────────────────────────────────────────
from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor

from .trellis_image_to_3d import TrellisVGGTTo3DPipeline


class TrellisHybridPipeline:
    """
    Hybrid 3-stage pipeline:

    1. SS      – uses ReconViaGen's TrellisVGGTTo3DPipeline (VGGT + DINOv2 features)
    2. Shape   – uses TRELLIS.2's Trellis2ImageTo3DPipeline  (shape_slat, DINOv3 features)
    3. Texture – uses TRELLIS.2's Trellis2ImageTo3DPipeline  (tex_slat,   DINOv3 features)

    The two sub-pipelines are kept separate so each can be moved between CPU/GPU
    independently when ``low_vram=True``.

    Args:
        vggt_pipeline   : Loaded TrellisVGGTTo3DPipeline  (provides SS components).
        trellis2_pipeline: Loaded Trellis2ImageTo3DPipeline (provides shape/tex slat + decode).
    """

    def __init__(
        self,
        vggt_pipeline: TrellisVGGTTo3DPipeline,
        trellis2_pipeline: Trellis2ImageTo3DPipeline,
        low_vram: bool = False,
    ):
        self.vggt_pipeline    = vggt_pipeline
        self.trellis2_pipeline = trellis2_pipeline
        self.low_vram = low_vram
        if low_vram:
            trellis2_pipeline.low_vram = True

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def _vggt_models_to(self, device) -> None:
        """Move vggt_pipeline inference models to *device* (for low-VRAM mode)."""
        vp = self.vggt_pipeline
        if hasattr(vp, 'VGGT_model') and vp.VGGT_model is not None:
            vp.VGGT_model.to(device)
        for model in vp.models.values():
            model.to(device)

    @property
    def device(self):
        return self.vggt_pipeline.device

    @property
    def pbr_attr_layout(self):
        return self.trellis2_pipeline.pbr_attr_layout

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess the input image — mirrors Trellis2ImageTo3DPipeline.preprocess_image:
        if the image already has a real alpha channel it is used directly;
        otherwise the background is removed with the rembg_model (BiRefNet).
        The foreground is then cropped and composited onto a black background.
        """
        t2p = self.trellis2_pipeline

        # Use existing alpha if present, otherwise run background removal
        has_alpha = False
        if image.mode == 'RGBA':
            alpha = np.array(image)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        max_size = max(image.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )

        if has_alpha:
            output = image
        else:
            image = image.convert('RGB')
            if t2p.low_vram:
                t2p.rembg_model.to(t2p.device)
            output = t2p.rembg_model(image)
            if t2p.low_vram:
                t2p.rembg_model.cpu()

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        return Image.fromarray((output * 255).astype(np.uint8))

    # ── Stage 1: SS via VGGT ─────────────────────────────────────────────────

    @torch.no_grad()
    def _run_ss_stage(
        self,
        images: List[Image.Image],
        target_ss_res: int,
        ss_sampler_params: dict,
    ) -> torch.Tensor:
        """
        Run ReconViaGen's VGGT-conditioned sparse-structure stage.

        Returns:
            coords : (N, 4) int tensor  [batch_idx, x, y, z]  in [0, target_ss_res)
        """
        vp = self.vggt_pipeline

        if self.low_vram:
            self._vggt_models_to(vp.device)

        # VGGT feature extraction
        with torch.cuda.amp.autocast(dtype=vp.VGGT_dtype):
            aggregated_tokens_list, _ = vp.vggt_feat(images)
        b, n, _, _ = aggregated_tokens_list[0].shape

        # DINOv2 image features (used by SS conditioning projector)
        image_cond = vp.encode_image(images).reshape(b, n, -1, 1024)

        # Build SS condition dict
        ss_cond = vp.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, 1)

        # Sample SS latent
        ss_flow = vp.models['sparse_structure_flow_model']
        reso    = ss_flow.resolution
        noise   = torch.randn(1, ss_flow.in_channels, reso, reso, reso).to(vp.device)

        merged_params = {**vp.sparse_structure_sampler_params, **ss_sampler_params}
        ss_latent = vp.sparse_structure_sampler.sample(
            ss_flow, noise, **ss_cond, **merged_params, verbose=True
        ).samples

        # Decode occupancy volume
        decoded = vp.models['sparse_structure_decoder'](ss_latent) > 0

        # Optionally pool to the resolution the downstream shape_slat model expects
        if target_ss_res != decoded.shape[2]:
            ratio   = decoded.shape[2] // target_ss_res
            decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        if self.low_vram:
            self._vggt_models_to('cpu')
            torch.cuda.empty_cache()

        return coords

    # ── Main run (single image) ───────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        images: List[Image.Image],
        seed: int = 42,
        ss_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        pipeline_type: str = '1024',
        preprocess_image: bool = True,
        return_latent: bool = False,
        max_num_tokens: int = 49152,
    ):
        """
        Run the full hybrid pipeline.

        Args:
            images                  : List of (preprocessed) PIL images.
            seed                    : Random seed.
            ss_sampler_params       : Override params for the SS sampler.
            shape_slat_sampler_params: Override params for the shape-SLat sampler.
            tex_slat_sampler_params : Override params for the tex-SLat sampler.
            pipeline_type           : '512' or '1024'  (controls which TRELLIS.2
                                      flow models are used for shape/tex slat).
            preprocess_image        : Whether to preprocess images first.
            return_latent           : If True, also return (shape_slat, tex_slat, res).

        Returns:
            List[MeshWithVoxel]  (one per sample, usually 1)
            optionally: (shape_slat, tex_slat, res)
        """
        assert pipeline_type in ('512', '1024', '1536'), \
            f"pipeline_type must be '512' or '1024' or '1536', got '{pipeline_type}'"

        torch.manual_seed(seed)

        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        # Target SS resolution: 32 for 512-pipeline, 64 for 1024-pipeline
        target_ss_res = {'512': 32, '1024': 32, '1536': 32}[pipeline_type]

        # ── Stage 1: SS ───────────────────────────────────────────────────────
        coords = self._run_ss_stage(images, target_ss_res, ss_sampler_params)

        # ── Stages 2 & 3: shape_slat + tex_slat (TRELLIS.2) ──────────────────
        t2p = self.trellis2_pipeline

        cond_512 = t2p.get_cond(images, 512)
        cond_1024 = t2p.get_cond(images, 1024) if pipeline_type != '512' else None

        if pipeline_type == '512':
            shape_slat = t2p.sample_shape_slat(
                cond_512,
                t2p.models['shape_slat_flow_model_512'],
                coords,
                shape_slat_sampler_params,
            )
            tex_slat = t2p.sample_tex_slat(
                cond_512,
                t2p.models['tex_slat_flow_model_512'],
                shape_slat,
                tex_slat_sampler_params,
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat, res = t2p.sample_shape_slat_cascade(
                cond_512, cond_1024,
                t2p.models['shape_slat_flow_model_512'], t2p.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords,
                shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = t2p.sample_tex_slat(
                cond_1024,
                t2p.models['tex_slat_flow_model_1024'],
                shape_slat,
                tex_slat_sampler_params,
            )
        elif pipeline_type == '1536':
            shape_slat, res = t2p.sample_shape_slat_cascade(
                cond_512, cond_1024,
                t2p.models['shape_slat_flow_model_512'], t2p.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = t2p.sample_tex_slat(
                cond_1024,
                t2p.models['tex_slat_flow_model_1024'],
                shape_slat,
                tex_slat_sampler_params,
            )

        # ── Stage 4: Decode ───────────────────────────────────────────────────
        torch.cuda.empty_cache()
        out_mesh = t2p.decode_latent(shape_slat, tex_slat, res)

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh

    # ── Multi-image run ───────────────────────────────────────────────────────

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        strategy: str = 'average_right',
        seed: int = 42,
        ss_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        pipeline_type: str = '1024',
        preprocess_image: bool = True,
        return_latent: bool = False,
        max_num_tokens: int = 49152,
    ):
        """
        Multi-image variant.

        The SS stage already uses all images jointly (VGGT processes them together).
        The shape/tex slat stages use TRELLIS.2's ``_multi_image_sample`` with
        the chosen fusion ``strategy``.

        For pipeline_type '1024' and '1536', a two-stage cascade is used:
          - LR: _multi_image_sample with conds_512 + shape_slat_flow_model_512
          - upsample via shape_slat_decoder.upsample
          - HR: _multi_image_sample with conds_1024 + shape_slat_flow_model_1024
        This mirrors the single-image run() cascade logic.

        Args:
            strategy: 'sequential' | 'average' | 'average_right' | 'weighted_average'
                      (passed to Trellis2ImageTo3DPipeline._multi_image_sample)
        """
        assert pipeline_type in ('512', '1024', '1536'), \
            f"pipeline_type must be '512', '1024', or '1536', got '{pipeline_type}'"

        # Single image → fall back to run()
        if len(images) == 1:
            return self.run(
                images, seed, ss_sampler_params,
                shape_slat_sampler_params, tex_slat_sampler_params,
                pipeline_type, preprocess_image, return_latent, max_num_tokens,
            )

        torch.manual_seed(seed)

        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        target_ss_res = {'512': 32, '1024': 32, '1536': 32}[pipeline_type]

        # ── Stage 1: SS (all images fed jointly to VGGT) ─────────────────────
        coords = self._run_ss_stage(images, target_ss_res, ss_sampler_params)

        # ── Per-image conditioning ────────────────────────────────────────────
        t2p = self.trellis2_pipeline
        conds_512  = [t2p.get_cond([img], 512) for img in images]
        conds_1024 = [t2p.get_cond([img], 1024) for img in images] if pipeline_type != '512' else None

        std_shape  = torch.tensor(t2p.shape_slat_normalization['std'])[None]
        mean_shape = torch.tensor(t2p.shape_slat_normalization['mean'])[None]
        std_tex    = torch.tensor(t2p.tex_slat_normalization['std'])[None]
        mean_tex   = torch.tensor(t2p.tex_slat_normalization['mean'])[None]

        def _sample_shape(fm, conds, tqdm_desc):
            noise_slat = SparseTensor(
                feats=torch.randn(coords.shape[0], fm.in_channels).to(t2p.device),
                coords=coords,
            )
            if t2p.low_vram:
                fm.to(t2p.device)
            slat = t2p._multi_image_sample(
                t2p.shape_slat_sampler, fm, noise_slat,
                conds, t2p.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc=tqdm_desc,
            )
            if t2p.low_vram:
                fm.cpu()
            return slat * std_shape.to(slat.device) + mean_shape.to(slat.device)

        def _sample_tex(fm, conds, shape_slat):
            s_norm = (shape_slat - mean_shape.to(shape_slat.device)) / std_shape.to(shape_slat.device)
            in_ch  = fm.in_channels if isinstance(fm, torch.nn.Module) else fm[0].in_channels
            noise_tex = s_norm.replace(
                feats=torch.randn(s_norm.coords.shape[0], in_ch - s_norm.feats.shape[1]).to(t2p.device)
            )
            if t2p.low_vram:
                fm.to(t2p.device)
            slat = t2p._multi_image_sample(
                t2p.tex_slat_sampler, fm, noise_tex,
                conds, t2p.tex_slat_sampler_params, tex_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling texture SLat",
                concat_cond=s_norm,
            )
            if t2p.low_vram:
                fm.cpu()
            return slat * std_tex.to(slat.device) + mean_tex.to(slat.device)

        if pipeline_type == '512':
            shape_slat = _sample_shape(t2p.models['shape_slat_flow_model_512'], conds_512, "Sampling shape SLat")
            tex_slat   = _sample_tex(t2p.models['tex_slat_flow_model_512'], conds_512, shape_slat)
            res = 512

        else:  # '1024' or '1536' — cascade: LR (512) → upsample → HR (1024)
            target_res = {'1024': 1024, '1536': 1536}[pipeline_type]

            # LR stage: multi-image sample at 512 resolution
            fm_lr = t2p.models['shape_slat_flow_model_512']
            noise_lr = SparseTensor(
                feats=torch.randn(coords.shape[0], fm_lr.in_channels).to(t2p.device),
                coords=coords,
            )
            if t2p.low_vram:
                fm_lr.to(t2p.device)
            slat_lr = t2p._multi_image_sample(
                t2p.shape_slat_sampler, fm_lr, noise_lr,
                conds_512, t2p.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling shape SLat (LR)",
            )
            if t2p.low_vram:
                fm_lr.cpu()
            slat_lr = slat_lr * std_shape.to(slat_lr.device) + mean_shape.to(slat_lr.device)

            # Upsample LR slat to HR coords
            if t2p.low_vram:
                t2p.models['shape_slat_decoder'].to(t2p.device)
                t2p.models['shape_slat_decoder'].low_vram = True
            hr_coords_raw = t2p.models['shape_slat_decoder'].upsample(slat_lr, upsample_times=4)
            if t2p.low_vram:
                t2p.models['shape_slat_decoder'].cpu()
                t2p.models['shape_slat_decoder'].low_vram = False

            hr_resolution = target_res
            while True:
                quant_coords = torch.cat([
                    hr_coords_raw[:, :1],
                    ((hr_coords_raw[:, 1:] + 0.5) / 512 * (hr_resolution // 16)).int(),
                ], dim=1)
                hr_c = quant_coords.unique(dim=0)
                if hr_c.shape[0] < max_num_tokens or hr_resolution == 1024:
                    if hr_resolution != target_res:
                        print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                    break
                hr_resolution -= 128

            # HR stage: multi-image sample at hr_resolution using conds_1024
            fm_hr = t2p.models['shape_slat_flow_model_1024']
            noise_hr = SparseTensor(
                feats=torch.randn(hr_c.shape[0], fm_hr.in_channels).to(t2p.device),
                coords=hr_c,
            )
            if t2p.low_vram:
                fm_hr.to(t2p.device)
            shape_slat = t2p._multi_image_sample(
                t2p.shape_slat_sampler, fm_hr, noise_hr,
                conds_1024, t2p.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling shape SLat (HR)",
            )
            if t2p.low_vram:
                fm_hr.cpu()
            shape_slat = shape_slat * std_shape.to(shape_slat.device) + mean_shape.to(shape_slat.device)
            res = hr_resolution

            tex_slat = _sample_tex(t2p.models['tex_slat_flow_model_1024'], conds_1024, shape_slat)

        # ── Stage 4: Decode ───────────────────────────────────────────────────
        torch.cuda.empty_cache()
        out_mesh = t2p.decode_latent(shape_slat, tex_slat, res)

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh

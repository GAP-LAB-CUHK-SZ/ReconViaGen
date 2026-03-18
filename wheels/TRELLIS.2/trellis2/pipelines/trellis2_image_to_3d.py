from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm as _tqdm
from .base import Pipeline
from . import samplers
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        from . import rembg as _rembg
        rembg_args = args['rembg_model']['args'].copy()
        rembg_args['model_name'] = 'ZhengPeng7/BiRefNet'
        pipeline.rembg_model = getattr(_rembg, args['rembg_model']['name'])(**rembg_args)
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    def _multi_image_sample(
        self,
        sampler,
        model,
        noise,
        conds: List[dict],
        base_params: dict,
        override_params: dict,
        strategy: str = 'sequential',
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **extra_kwargs,
    ):
        """
        Custom denoising loop that supports multiple conditioning images.

        Strategy 'sequential': at each denoising step i, use conds[i % N] as the condition.
        Strategy 'average':    at each denoising step, run the model once per image and
                               average the resulting pred_x_prev tensors.

        Args:
            sampler:         The sampler instance (FlowEulerCfgSampler, etc.).
            model:           The flow model.
            noise:           Initial noise (dense torch.Tensor or SparseTensor).
            conds:           List of per-image condition dicts {'cond': ..., 'neg_cond': ...}.
            base_params:     Default sampler params from the pipeline config.
            override_params: Per-call override params (e.g. from app sliders).
            strategy:        'sequential' or 'average'.
            extra_kwargs:    Extra kwargs forwarded to sample_once (e.g. concat_cond).
        """
        all_params = {**base_params, **override_params}
        steps = all_params.pop('steps', 50)
        rescale_t = all_params.pop('rescale_t', 1.0)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

        sample = noise
        n = len(conds)

        # 对 average_right / weighted_average / adaptive_guidance_weight / fixed_guidance_rescale 策略，
        # 在循环前提取 CFG 参数。CFG 专属 key 不应透传给底层原始模型。
        _CFG_KEYS = {'guidance_strength', 'guidance_rescale', 'guidance_interval'}

        if strategy in ('average_right', 'weighted_average', 'adaptive_guidance_weight', 'fixed_guidance_rescale'):
            guidance_strength = all_params.get('guidance_strength', 3.0)
            guidance_rescale  = all_params.get('guidance_rescale', 0.0)
            neg_cond          = conds[0]['neg_cond']
            # 去掉 CFG 专属参数，剩余部分透传给原始模型
            raw_kw = {k: v for k, v in all_params.items() if k not in _CFG_KEYS}

            if strategy == 'average_right':
                # 均等权重（在循环外预计算，循环内直接用）
                w = torch.full((n,), 1.0 / n)
            # 'weighted_average' / 'adaptive_guidance_weight':
            # 权重在每个去噪步骤内按 token 动态计算，此处无需预计算

        for step_idx, (t, t_prev) in enumerate(_tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)):
            if strategy == 'sequential':
                c = conds[step_idx % n]
                out = sampler.sample_once(
                    model, sample, t, t_prev,
                    c['cond'],
                    neg_cond=c['neg_cond'],
                    **all_params,
                    **extra_kwargs,
                )
                sample = out.pred_x_prev

            elif strategy == 'average':
                x_prevs = []
                for c in conds:
                    out = sampler.sample_once(
                        model, sample, t, t_prev,
                        c['cond'],
                        neg_cond=c['neg_cond'],
                        **all_params,
                        **extra_kwargs,
                    )
                    x_prevs.append(out.pred_x_prev)
                if isinstance(x_prevs[0], SparseTensor):
                    avg_feats = torch.stack([x.feats for x in x_prevs]).mean(dim=0)
                    sample = x_prevs[0].replace(avg_feats)
                else:
                    sample = torch.stack(x_prevs).mean(dim=0)

            else:  # 'average_right' | 'weighted_average' | 'adaptive_guidance_weight' | 'fixed_guidance_rescale'
                dev = sample.feats.device if isinstance(sample, SparseTensor) else sample.device
                v_final = None  # may be set directly by 'fixed_guidance_rescale'

                # 1次 uncond 调用 + N次 cond 调用（直接调用底层 FlowEulerSampler，绕过 CFG mixin）
                v_neg = samplers.FlowEulerSampler._inference_model(
                    sampler, model, sample, t, neg_cond, **raw_kw, **extra_kwargs
                )
                v_pos_list = [
                    samplers.FlowEulerSampler._inference_model(
                        sampler, model, sample, t, c['cond'], **raw_kw, **extra_kwargs
                    )
                    for c in conds
                ]

                # 计算加权平均条件 velocity
                if strategy == 'average_right':
                    # 均等权重
                    wt = w.to(dev)
                    if isinstance(v_pos_list[0], SparseTensor):
                        stacked   = torch.stack([v.feats for v in v_pos_list], dim=0)  # (N, nnz, C)
                        avg_feats = (wt[:, None, None] * stacked).sum(0)               # (nnz, C)
                        v_pos_avg = v_pos_list[0].replace(avg_feats)
                    else:
                        stacked   = torch.stack(v_pos_list, dim=0)                     # (N, B, C, ...)
                        v_pos_avg = (wt.view(n, *([1] * (stacked.ndim - 1))) * stacked).sum(0)

                elif strategy == 'weighted_average':
                    # Per-token velocity-deviation weights:
                    # 各视角 velocity 偏离跨视角均值越小，该视角在该 token 上的权重越高。
                    # 被遮挡或高光的视角在对应区域 velocity 偏差大，权重被局部压低。
                    if isinstance(v_pos_list[0], SparseTensor):
                        stacked   = torch.stack([v.feats for v in v_pos_list], dim=0)  # (N, nnz, C)
                        mean_v    = stacked.mean(dim=0)                                # (nnz, C)
                        dev_norm  = (stacked - mean_v[None]).norm(dim=2)               # (N, nnz)
                        w_tok     = F.softmax(-dev_norm, dim=0)                        # (N, nnz)
                        avg_feats = (w_tok[:, :, None] * stacked).sum(0)              # (nnz, C)
                        v_pos_avg = v_pos_list[0].replace(avg_feats)
                    else:
                        stacked   = torch.stack(v_pos_list, dim=0)                    # (N, B, C, ...)
                        mean_v    = stacked.mean(dim=0)                               # (B, C, ...)
                        dev_norm  = (stacked - mean_v[None]).pow(2).sum(dim=2).sqrt() # (N, B, ...)
                        w_tok     = F.softmax(-dev_norm, dim=0)                       # (N, B, ...)
                        v_pos_avg = (w_tok.unsqueeze(2) * stacked).sum(0)             # (B, C, ...)

                elif strategy == 'adaptive_guidance_weight':
                    # Per-token guidance magnitude weights with t-adaptive temperature.
                    #
                    # 权重依据：|v_cond_i - v_uncond| 是模型对该 token 在视角 i 下的置信度代理。
                    # 若视角 i 可见该区域，模型给出强烈的条件偏移（幅度大）→ 权重高；
                    # 若视角 i 遮挡该区域，模型退回 uncond（幅度小）→ 权重低。
                    # 不依赖跨视角均值，因此不受 outlier 视角污染。
                    #
                    # 温度 τ = t（当前噪声水平）：
                    #   t ≈ 1（早期步）→ 高温 → 权重趋于均等，稳定全局结构；
                    #   t ≈ 0（晚期步）→ 低温 → 权重向高 magnitude 视角集中，精化局部细节。
                    if isinstance(v_pos_list[0], SparseTensor):
                        stacked   = torch.stack([v.feats for v in v_pos_list], dim=0)  # (N, nnz, C)
                        v_neg_f   = v_neg.feats                                        # (nnz, C)
                        mag       = (stacked - v_neg_f[None]).norm(dim=2)              # (N, nnz)
                        w_tok     = F.softmax(mag / (t + 1e-3), dim=0)                # (N, nnz)
                        avg_feats = (w_tok[:, :, None] * stacked).sum(0)              # (nnz, C)
                        v_pos_avg = v_pos_list[0].replace(avg_feats)
                    else:
                        stacked   = torch.stack(v_pos_list, dim=0)                    # (N, B, C, ...)
                        mag       = (stacked - v_neg[None]).pow(2).sum(dim=2).sqrt()  # (N, B, ...)
                        w_tok     = F.softmax(mag / (t + 1e-3), dim=0)               # (N, B, ...)
                        v_pos_avg = (w_tok.unsqueeze(2) * stacked).sum(0)             # (B, C, ...)

                else:  # 'fixed_guidance_rescale'
                    # Theoretically correct PoE with per-view guidance rescale.
                    #
                    # 'average_right' + rescale 的问题：先对各视角 cond velocity 取平均，再计算
                    # std(x_0_avg) 作为 rescale 参考。当视角之间不一致时，平均会抹平差异，导致
                    # std(x_0_avg) 远小于各视角自身的 std(x_0_cond_i)，从而使 rescale 修正量
                    # 被低估（under-rescale），guidance 效果被削弱。
                    #
                    # 修正方案：对每个视角 i 独立执行 CFG（guidance_strength/N）+ rescale，
                    # 以该视角自身的 cond 预测作为 std 参考，再将各视角的修正贡献叠加：
                    #
                    #   v_final = v_neg + Σ_i (v_i_rescaled - v_neg)
                    #           = Σ_i v_i_rescaled - (N-1) * v_neg
                    #
                    # 当 guidance_rescale=0 或各视角完全一致时，退化为 'average_right'。
                    if guidance_strength == 0:
                        v_final = v_neg
                    else:
                        gs_per     = guidance_strength / n
                        is_sparse  = isinstance(v_pos_list[0], SparseTensor)
                        v_i_list   = []

                        for v_cond_i in v_pos_list:
                            # per-view CFG: 每个视角贡献 gs/N 的 guidance 强度
                            if is_sparse:
                                cfg_feats = gs_per * v_cond_i.feats + (1.0 - gs_per) * v_neg.feats
                                v_cfg_i   = v_cond_i.replace(cfg_feats)
                            else:
                                v_cfg_i   = gs_per * v_cond_i + (1.0 - gs_per) * v_neg

                            if guidance_rescale > 0:
                                x0_cond_i = sampler._pred_to_xstart(sample, t, v_cond_i)
                                x0_cfg_i  = sampler._pred_to_xstart(sample, t, v_cfg_i)

                                if is_sparse:
                                    # 用该视角自身 cond 预测的 std 作为修正参考（全局标量）
                                    std_cond  = x0_cond_i.feats.std().clamp(min=1e-8)
                                    std_cfg   = x0_cfg_i.feats.std().clamp(min=1e-8)
                                    ratio     = (std_cond / std_cfg).clamp(max=2.0)
                                    # x_0_i = gr * (x0_cfg * ratio) + (1-gr) * x0_cfg
                                    #        = x0_cfg * (gr * ratio + 1 - gr)
                                    x0_feats  = (guidance_rescale * ratio + (1.0 - guidance_rescale)) * x0_cfg_i.feats
                                    x0_i      = x0_cfg_i.replace(x0_feats)
                                else:
                                    d         = list(range(1, x0_cond_i.ndim))
                                    std_cond  = x0_cond_i.std(dim=d, keepdim=True).clamp(min=1e-8)
                                    std_cfg   = x0_cfg_i.std(dim=d, keepdim=True).clamp(min=1e-8)
                                    ratio     = (std_cond / std_cfg).clamp(max=2.0)
                                    x0_i      = (guidance_rescale * ratio + (1.0 - guidance_rescale)) * x0_cfg_i

                                v_i_list.append(sampler._xstart_to_pred(sample, t, x0_i))
                            else:
                                v_i_list.append(v_cfg_i)

                        # PoE 聚合：v_final = Σ_i v_i - (N-1) * v_neg
                        if is_sparse:
                            sum_feats = torch.stack([v.feats for v in v_i_list], dim=0).sum(0)
                            v_final   = v_i_list[0].replace(sum_feats - (n - 1) * v_neg.feats)
                        else:
                            v_final   = torch.stack(v_i_list, dim=0).sum(0) - (n - 1) * v_neg

                # 对平均后的条件 velocity 统一应用一次 CFG（fixed_guidance_rescale 已直接设置 v_final，跳过）
                if v_final is None:
                    if guidance_strength == 1:
                        v_final = v_pos_avg
                    elif guidance_strength == 0:
                        v_final = v_neg
                    else:
                        v_final = guidance_strength * v_pos_avg + (1.0 - guidance_strength) * v_neg

                        # CFG rescale：基于平均后的 pred_pos 计算 std（修正旧实现按单图计算的问题）
                        if guidance_rescale > 0:
                            x_0_pos = sampler._pred_to_xstart(sample, t, v_pos_avg)
                            x_0_cfg = sampler._pred_to_xstart(sample, t, v_final)
                            std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                            std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                            x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                            x_0 = guidance_rescale * x_0_rescaled + (1.0 - guidance_rescale) * x_0_cfg
                            v_final = sampler._xstart_to_pred(sample, t, x_0)

                # Euler 步
                sample = sample - (t - t_prev) * v_final

        return sample

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        strategy: str = 'sequential',
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline conditioned on multiple images of the same object.

        Four fusion strategies are supported:

        * ``'sequential'``: at denoising step *i*, the image ``images[i % N]`` is used as
          the condition.  Each step sees only one view.

        * ``'average'``: at every denoising step the model is evaluated once per image and
          the resulting ``pred_x_prev`` tensors are averaged.  N× more expensive per step.

        * ``'average_right'``: correct multi-image CFG (Product-of-Experts).  1 uncond call
          + N cond calls; conditional velocities are equally averaged then CFG (including
          rescale) is applied *once* on the average — fixing the per-image rescale bias of
          ``'average'``.  Same cost as ``'average'`` (N+1 forward passes).

        * ``'weighted_average'``: same as ``'average_right'`` (1 uncond + N cond calls, CFG
          applied once on the blend) but uses **per-token** weights computed dynamically at
          every denoising step.  For each latent token, the L2 distance of each view's
          velocity to the cross-view mean velocity is computed; views that deviate less
          (agree with the consensus for that surface region) receive higher weight via
          softmax(-deviation).  Views suffering from occlusion or specularity are suppressed
          locally rather than globally, while well-observed regions are still smoothly blended.

        * ``'adaptive_guidance_weight'``: same cost as ``'average_right'`` (1 uncond + N cond
          calls).  For each latent token independently, the per-view weight is proportional to
          the **guidance magnitude** ``‖v_cond_i − v_uncond‖`` at that token — a direct proxy
          for how much the model "cares about" (i.e. can see) the corresponding surface region
          given view *i*.  Occluded or specular views naturally produce small guidance signals
          and are suppressed without relying on a cross-view mean that could itself be
          corrupted by outlier views.  The softmax temperature is set to the current noise
          level *t*: at early steps (t≈1) weights are nearly uniform for stable global
          structure; at late steps (t≈0) the temperature sharpens to let the most confident
          views dominate fine-detail refinement.

        * ``'fixed_guidance_rescale'``: theoretically correct Product-of-Experts CFG with
          **per-view independent rescale**.  ``'average_right'`` first averages the per-view
          cond velocities and then applies guidance rescale once using ``std(x_0_avg)`` as the
          reference — when views disagree this average suppresses variance, making the rescale
          reference too small and under-correcting the guidance amplification.  This strategy
          instead applies CFG (strength ``gs/N``) and rescale *independently* for each view,
          using that view's own ``std(x_0_cond_i)`` as the reference, and then aggregates via
          the PoE sum ``v_final = Σ_i v_i_rescaled − (N−1)·v_uncond``.  Degrades exactly to
          ``'average_right'`` when ``guidance_rescale=0`` or all views agree.

        Args:
            images:   One or more images of the same object (raw or pre-processed).
            strategy: ``'sequential'`` | ``'average'`` | ``'average_right'`` |
                      ``'weighted_average'`` | ``'adaptive_guidance_weight'`` |
                      ``'fixed_guidance_rescale'``.
            (remaining args identical to :meth:`run`)
        """
        assert strategy in (
            'sequential', 'average', 'average_right',
            'weighted_average', 'adaptive_guidance_weight', 'fixed_guidance_rescale',
        ), (
            f"Unknown strategy '{strategy}'. Choose 'sequential', 'average', "
            f"'average_right', 'weighted_average', 'adaptive_guidance_weight', "
            f"or 'fixed_guidance_rescale'."
        )
        assert len(images) > 0, "At least one image is required."

        # If only a single image is provided fall back to the standard pipeline.
        if len(images) == 1:
            return self.run(
                images[0], num_samples, seed,
                sparse_structure_sampler_params,
                shape_slat_sampler_params,
                tex_slat_sampler_params,
                preprocess_image, return_latent, pipeline_type, max_num_tokens,
            )

        # ── Pipeline-type validation (mirrors run()) ──────────────────────────
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'tex_slat_flow_model_512' in self.models
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # ── Preprocessing ─────────────────────────────────────────────────────
        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]
        torch.manual_seed(seed)

        # ── Per-image conditioning ────────────────────────────────────────────
        conds_512 = [self.get_cond([img], 512) for img in images]
        conds_1024 = [self.get_cond([img], 1024) for img in images] \
            if pipeline_type != '512' else None

        # ── Normalization constants ───────────────────────────────────────────
        std_shape  = torch.tensor(self.shape_slat_normalization['std'])[None]
        mean_shape = torch.tensor(self.shape_slat_normalization['mean'])[None]
        std_tex    = torch.tensor(self.tex_slat_normalization['std'])[None]
        mean_tex   = torch.tensor(self.tex_slat_normalization['mean'])[None]

        # ── Stage 1: Sparse Structure ─────────────────────────────────────────
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        flow_model_ss = self.models['sparse_structure_flow_model']
        noise = torch.randn(
            num_samples, flow_model_ss.in_channels,
            flow_model_ss.resolution, flow_model_ss.resolution, flow_model_ss.resolution,
        ).to(self.device)
        if self.low_vram:
            flow_model_ss.to(self.device)
        z_s = self._multi_image_sample(
            self.sparse_structure_sampler, flow_model_ss, noise,
            conds_512, self.sparse_structure_sampler_params, sparse_structure_sampler_params,
            strategy=strategy, verbose=True, tqdm_desc="Sampling sparse structure",
        )
        if self.low_vram:
            flow_model_ss.cpu()

        decoder_ss = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder_ss.to(self.device)
        decoded = decoder_ss(z_s) > 0
        if self.low_vram:
            decoder_ss.cpu()
        if ss_res != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_res
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        # ── Stages 2 & 3: Shape SLat + Texture SLat ──────────────────────────
        def _sample_shape(fm, conds, tqdm_desc):
            noise_slat = SparseTensor(
                feats=torch.randn(coords.shape[0], fm.in_channels).to(self.device),
                coords=coords,
            )
            if self.low_vram:
                fm.to(self.device)
            slat = self._multi_image_sample(
                self.shape_slat_sampler, fm, noise_slat,
                conds, self.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc=tqdm_desc,
            )
            if self.low_vram:
                fm.cpu()
            return slat * std_shape.to(slat.device) + mean_shape.to(slat.device)

        def _sample_tex(fm, conds, shape_slat):
            s_norm = (shape_slat - mean_shape.to(shape_slat.device)) / std_shape.to(shape_slat.device)
            in_ch = fm.in_channels if isinstance(fm, nn.Module) else fm[0].in_channels
            noise_tex = s_norm.replace(
                feats=torch.randn(s_norm.coords.shape[0], in_ch - s_norm.feats.shape[1]).to(self.device)
            )
            if self.low_vram:
                fm.to(self.device)
            slat = self._multi_image_sample(
                self.tex_slat_sampler, fm, noise_tex,
                conds, self.tex_slat_sampler_params, tex_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling texture SLat",
                concat_cond=s_norm,
            )
            if self.low_vram:
                fm.cpu()
            return slat * std_tex.to(slat.device) + mean_tex.to(slat.device)

        if pipeline_type == '512':
            shape_slat = _sample_shape(self.models['shape_slat_flow_model_512'], conds_512, "Sampling shape SLat")
            tex_slat   = _sample_tex(self.models['tex_slat_flow_model_512'], conds_512, shape_slat)
            res = 512

        elif pipeline_type == '1024':
            shape_slat = _sample_shape(self.models['shape_slat_flow_model_1024'], conds_1024, "Sampling shape SLat")
            tex_slat   = _sample_tex(self.models['tex_slat_flow_model_1024'], conds_1024, shape_slat)
            res = 1024

        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            target_res = 1024 if pipeline_type == '1024_cascade' else 1536

            # LR shape slat
            fm_lr = self.models['shape_slat_flow_model_512']
            noise_lr = SparseTensor(
                feats=torch.randn(coords.shape[0], fm_lr.in_channels).to(self.device),
                coords=coords,
            )
            if self.low_vram:
                fm_lr.to(self.device)
            slat_lr = self._multi_image_sample(
                self.shape_slat_sampler, fm_lr, noise_lr,
                conds_512, self.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling shape SLat (LR)",
            )
            if self.low_vram:
                fm_lr.cpu()
            slat_lr = slat_lr * std_shape.to(slat_lr.device) + mean_shape.to(slat_lr.device)

            # Upsample
            if self.low_vram:
                self.models['shape_slat_decoder'].to(self.device)
                self.models['shape_slat_decoder'].low_vram = True
            hr_coords_raw = self.models['shape_slat_decoder'].upsample(slat_lr, upsample_times=4)
            if self.low_vram:
                self.models['shape_slat_decoder'].cpu()
                self.models['shape_slat_decoder'].low_vram = False

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

            # HR shape slat
            fm_hr = self.models['shape_slat_flow_model_1024']
            noise_hr = SparseTensor(
                feats=torch.randn(hr_c.shape[0], fm_hr.in_channels).to(self.device),
                coords=hr_c,
            )
            if self.low_vram:
                fm_hr.to(self.device)
            shape_slat = self._multi_image_sample(
                self.shape_slat_sampler, fm_hr, noise_hr,
                conds_1024, self.shape_slat_sampler_params, shape_slat_sampler_params,
                strategy=strategy, verbose=True, tqdm_desc="Sampling shape SLat (HR)",
            )
            if self.low_vram:
                fm_hr.cpu()
            shape_slat = shape_slat * std_shape.to(shape_slat.device) + mean_shape.to(shape_slat.device)
            res = hr_resolution

            tex_slat = _sample_tex(self.models['tex_slat_flow_model_1024'], conds_1024, shape_slat)

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

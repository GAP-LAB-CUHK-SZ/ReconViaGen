from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import trimesh
import os
import random
import trellis.modules.sparse as sp
from trellis.models.sparse_structure_vae import *
from contextlib import contextmanager

import sys
sys.path.append("wheels/vggt")
from wheels.vggt.vggt.models.vggt import VGGT
from typing import *
from scipy.spatial.transform import Rotation
from transformers import AutoModelForImageSegmentation
import rembg
# for app_refine.py, please uncomment these lines
from dreamsim import dreamsim 
from tqdm import tqdm

def export_point_cloud(xyz, color):
    # Convert tensors to numpy arrays if needed
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    
    color = (color * 255).astype(np.uint8)

    # Create point cloud using trimesh
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=color)
    
    return point_cloud

def normalize_trimesh(mesh):
    # Calculate the mesh centroid and bounding box extents
    centroid = mesh.centroid
    # Determine the scale based on the largest extent to fit into unit cube
    # Normalizing: Center and scale the vertices
    mesh.vertices -= centroid

    extents = mesh.extents
    scale = max(extents)
    mesh.vertices /= scale

    return mesh

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

from scipy.ndimage import binary_dilation
def voxelize_trimesh(mesh, resolution=(64, 64, 64), stride=4):
    """
    Voxelize a given trimesh object with the specified resolution, incorporating 4x anti-aliasing.
    First voxelizes at a 4x resolution and then downsamples to the target resolution.

    Args:
        mesh (trimesh.Trimesh): The input trimesh object to be voxelized.
        resolution (tuple): The voxel grid resolution as (x, y, z). Default is (64, 64, 64).

    Returns:
        np.ndarray: A boolean numpy array representing the voxel grid where True indicates
                    the presence of the mesh in that voxel and False otherwise.
    """
    target_density = max(resolution)
    target_edge_length = 1.0 / target_density
    max_edge_for_subdivision = target_edge_length / 2  

    # Calculate the higher resolution for 4x anti-aliasing
    anti_aliasing_density = target_density * stride
    anti_aliasing_edge_length = 1.0 / anti_aliasing_density
    anti_aliasing_max_edge_for_subdivision = anti_aliasing_edge_length / 2  

    # Get the vertices and faces of the mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # Subdivide the mesh for the higher resolution voxelization
    try:
        new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
            vertices, faces, anti_aliasing_max_edge_for_subdivision
        )
        subdivided_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    except Exception as e:
        print(f"Unexpected error during mesh subdivision for anti-aliasing: {e}")
        raise

    # Voxelize the subdivided mesh at the higher resolution
    try:
        high_res_voxel_grid = subdivided_mesh.voxelized(
            pitch=anti_aliasing_edge_length, method="binvox", exact=True
        )
    except:
        print("Voxelization using 'binvox' method failed for anti-aliasing")
        high_res_voxel_grid = subdivided_mesh.voxelized(pitch=anti_aliasing_edge_length)
        print("Falling back to default voxelization method for anti-aliasing.")
    high_res_boolean_array = high_res_voxel_grid.matrix.astype(bool)

    x_stride, y_stride, z_stride = [int(anti_aliasing_density / target_density)] * 3
    downsampled_shape = (
        high_res_boolean_array.shape[0] // x_stride,
        high_res_boolean_array.shape[1] // y_stride,
        high_res_boolean_array.shape[2] // z_stride
    )
    downsampled_array = np.zeros(downsampled_shape, dtype=bool)

    # Use NumPy's strided tricks to efficiently access sub-cubes for downsampling
    shape = (downsampled_shape[0], downsampled_shape[1], downsampled_shape[2], x_stride, y_stride, z_stride)
    strides = (x_stride * high_res_boolean_array.strides[0],
               y_stride * high_res_boolean_array.strides[1],
               z_stride * high_res_boolean_array.strides[2],
               high_res_boolean_array.strides[0],
               high_res_boolean_array.strides[1],
               high_res_boolean_array.strides[2])
    sub_cubes = np.lib.stride_tricks.as_strided(high_res_boolean_array, shape=shape, strides=strides)
    downsampled_array = np.any(sub_cubes, axis=(3, 4, 5))

    return downsampled_array

def get_occupied_coordinates(voxel_grid):
    # Find the indices of occupied voxels
    occupied_indices = np.argwhere(voxel_grid)
    
    coords = torch.tensor(occupied_indices, dtype=torch.int8)  # Use float for scaling operations
    
    # Add a leading dimension for batch size or any additional data associations
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords + 1], dim=1)

    # Move to GPU if required
    coords = coords.to('cuda:0')
    
    return coords

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    default_image_resolution = 518
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        try:
            dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), name, source='local',pretrained=True)
        except:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image, resolution=518, no_background=True, recenter=True) -> Image.Image:
        """
        Preprocess the input image using BiRefNet for background removal.
        Includes padding to maintain aspect ratio when resizing to 518x518.
        """
        # if has alpha channel, use it directly
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, -1]
            if not np.all(alpha == 255):
                has_alpha = True
        
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
                        
            # Get mask using BiRefNet
            mask = self._get_birefnet_mask(input)
            
            # Convert input to RGBA and apply mask
            input_rgba = input.convert('RGBA')
            input_array = np.array(input_rgba)
            input_array[:, :, 3] = mask * 255  # Apply mask to alpha channel
            output = Image.fromarray(input_array)

        # Process the output image
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:  # Handle case where no foreground is detected
            return input.convert('RGB')
        
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        height, width = alpha.shape
        if not recenter:
            center = [width / 2, height / 2]
            size = max(bbox[2] - bbox[0], 
                       bbox[3] - bbox[1], 
                       (bbox[2] - width / 2) * 2, 
                       (width / 2 - bbox[0]) * 2, 
                       (height / 2 - bbox[1]) * 2, 
                       (bbox[3] - height / 2) * 2)
            
        
        # Calculate and apply crop bbox
        if not no_background:
            if height > width:
                center[0] = width / 2
                if center[1] < width / 2:
                    center[1] = width / 2
                elif center[1] > height - width / 2:
                    center[1] = height - width / 2
            else:
                center[1] = height / 2
                if center[0] < height / 2:
                    center[0] = height / 2
                elif center[0] > width - height / 2:
                    center[0] = width - height / 2

            size = min(center[0], center[1], input.width - center[0], input.height - center[1], size) * 2

        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2)
        )
        
        # Ensure bbox is within image bounds
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(output.width, bbox[2]),
            min(output.height, bbox[3])
        )
        
        output = output.crop(bbox)
        
        # Add padding to maintain aspect ratio
        width, height = output.size
        if width > height:
            new_height = width
            padding = (width - height) // 2
            padded_output = Image.new('RGBA', (width, new_height), (0, 0, 0, 0))
            padded_output.paste(output, (0, padding))
        else:
            new_width = height
            padding = (height - width) // 2
            padded_output = Image.new('RGBA', (new_width, height), (0, 0, 0, 0))
            padded_output.paste(output, (padding, 0))
        
        # Resize padded image to target size
        # padded_output = padded_output.resize((resolution, resolution), Image.Resampling.LANCZOS)
        padded_output = torch.from_numpy(np.array(padded_output).astype(np.float32)) / 255
        padded_output = F.interpolate(padded_output.unsqueeze(0).permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        
        # Final processing
        output = padded_output.cpu().numpy()
        if no_background:
            output = np.dstack((
                output[:, :, :3] * (output[:, :, 3:4] > 0.8),  # RGB channels premultiplied by alpha
                output[:, :, 3]                         # Original alpha channel
            ))
        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        
        return output

    def _get_birefnet_mask(self, image: Image.Image) -> np.ndarray:
        """Get object mask using BiRefNet"""
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        return (mask_np > 128).astype(np.uint8)

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]], w_layernorm=True) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            image = F.interpolate(image, self.default_image_resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.default_image_resolution, self.default_image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        if getattr(self, 'low_vram', False):
            self.models['image_cond_model'].to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        if getattr(self, 'low_vram', False):
            self.models['image_cond_model'].cpu()
            torch.cuda.empty_cache()
        if w_layernorm:
            features = F.layer_norm(features, features.shape[-1:])
        return features
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        if noise is None:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if getattr(self, 'low_vram', False):
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        if getattr(self, 'low_vram', False):
            flow_model.cpu()

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        if getattr(self, 'low_vram', False):
            decoder.to(self.device)
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        if getattr(self, 'low_vram', False):
            decoder.cpu()
            torch.cuda.empty_cache()

        return coords
    
    def sample_sparse_structure_opt(
        self,
        cond: dict,
        ss: torch.Tensor,
        ss_learning_rate: float=1e-1, 
        ss_start_t: float=0.6,
        num_samples: int = 1,
        sampler_params: dict = {},
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        ss_decoder = self.models['sparse_structure_decoder']
        if getattr(self, 'low_vram', False):
            flow_model.to(self.device)
            ss_decoder.to(self.device)
        ss = ss.float()
        reso = flow_model.resolution
        if noise is None:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample_ss_opt_delta_v(
            flow_model,
            ss_decoder,
            ss_learning_rate,
            ss_start_t,
            ss,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        if getattr(self, 'low_vram', False):
            flow_model.cpu()
            ss_decoder.cpu()
            torch.cuda.empty_cache()

        return coords

    def sample_sparse_structure_opt_noise(
        self,
        cond: dict,
        ss: torch.Tensor,
        ss_learning_rate: float=1e-3, 
        num_samples: int = 1,
        sampler_params: dict = {},
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        if getattr(self, 'low_vram', False):
            flow_model.to(self.device)
            self.models['sparse_structure_decoder'].to(self.device)
        ss = ss.float()
        reso = flow_model.resolution
        if noise is None:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        torch.cuda.empty_cache()
        noise = torch.nn.Parameter(noise.to(self.device))
        optimizer = torch.optim.Adam([noise], betas=(0.5, 0.9), lr=ss_learning_rate)
        total_steps = 5
        def cosine_anealing(step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        fix_cond = cond['cond'].clone()
        with tqdm(total=total_steps, disable=False, desc='Geometry (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                shuffle_idx = torch.randperm(fix_cond.shape[0])
                cond['cond'] = fix_cond[shuffle_idx]
                norm_noise = (noise - noise.mean()) / noise.std()
                ss_slat = self.sparse_structure_sampler.sample_opt(
                    flow_model,
                    norm_noise,
                    **cond,
                    **{**self.sparse_structure_sampler_params, **{"steps": 1, "cfg_strength": sampler_params["cfg_strength"]}},
                    verbose=False
                ).samples
                ss_decoder = self.models['sparse_structure_decoder']
                logits = F.sigmoid(ss_decoder(ss_slat))
                loss = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
                # loss.backward()
                # optimizer.step()
                # 仅对 noise 求导，避免保留整个计算图（比 retain_graph=True 更省显存）
                grads = torch.autograd.grad(loss, noise, retain_graph=False, allow_unused=False)[0]
                # 把梯度写回 noise.grad 供 optimizer 使用
                noise.grad = grads
                optimizer.step()
                optimizer.param_groups[0]['lr'] = cosine_anealing(step, total_steps, ss_learning_rate, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        
        noise = noise.detach()
        torch.cuda.empty_cache()
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        coords = torch.argwhere(ss_decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        if getattr(self, 'low_vram', False):
            flow_model.cpu()
            self.models['sparse_structure_decoder'].cpu()
            torch.cuda.empty_cache()
        return coords

    def encode_slat(
        self,
        slat: sp.SparseTensor,
    ):
        ret = {}
        slat = self.models['slat_encoder'](slat, sample_posterior=False)
        ret['slat'] = slat
        return ret

    @torch.no_grad()
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        ret['slat'] = slat
        if 'gaussian' in formats:
            if getattr(self, 'low_vram', False):
                self.models['slat_decoder_gs'].to(self.device)
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
            if getattr(self, 'low_vram', False):
                self.models['slat_decoder_gs'].cpu()
                torch.cuda.empty_cache()
        if 'mesh' in formats:
            if getattr(self, 'low_vram', False):
                self.models['slat_decoder_mesh'].to(self.device)
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
            if getattr(self, 'low_vram', False):
                self.models['slat_decoder_mesh'].cpu()
                torch.cuda.empty_cache()
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        if getattr(self, 'low_vram', False):
            flow_model.to(self.device)
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        if getattr(self, 'low_vram', False):
            flow_model.cpu()
            torch.cuda.empty_cache()

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    def sample_slat_opt(
        self,
        apperance_learning_rate,
        start_t,
        input_images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        slat_decoder_gs = self.models['slat_decoder_gs']
        slat_decoder_mesh = self.models['slat_decoder_mesh']
        if getattr(self, 'low_vram', False):
            flow_model.to(self.device)
            slat_decoder_gs.to(self.device)
            slat_decoder_mesh.to(self.device)
            self.dreamsim_model.to(self.device)
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample_slat_opt_delta_v(
            flow_model,
            slat_decoder_gs,
            slat_decoder_mesh,
            std,
            mean,
            self.dreamsim_model,
            apperance_learning_rate,
            start_t,
            input_images,
            extrinsics, 
            intrinsics,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        if getattr(self, 'low_vram', False):
            flow_model.cpu()
            slat_decoder_gs.cpu()
            slat_decoder_mesh.cpu()
            self.dreamsim_model.cpu()
            torch.cuda.empty_cache()

        slat = slat * std + mean
        # from trellis.utils import render_utils, postprocessing_utils
        # import imageio
        # std = torch.tensor(self.slat_normalization['std'])[None].to(noise.device)
        # mean = torch.tensor(self.slat_normalization['mean'])[None].to(noise.device)
        # for i in range(sampler_params['steps']):
        #     latent = slat.pred_x_0[i] * std + mean
        #     outputs = self.decode_slat(latent, ["mesh", "gaussian"])
        #     video_geo = render_utils.render_video(outputs['mesh'][0], resolution=512, pitch=0, inverse_direction=True, num_frames=120)['normal']
        #     video_color = render_utils.render_video(outputs['gaussian'][0], resolution=512, pitch=0, inverse_direction=True, num_frames=120)['color']
        #     video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
        #     imageio.mimsave('outputs/slat_iter_{i:02d}.mp4'.format(i=i), video, fps=15)
        return slat

    def get_input(self, batch_data):
        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)

        images = batch_data['source_image']
        cond = self.encode_image(images)
        if random.random() > 0.5:
            cond = torch.zeros_like(cond)

        target_feats = batch_data['target_feats']
        target_coords = batch_data['target_coords']
        targets = sp.SparseTensor(target_feats, target_coords).to(self.device)
        targets = (targets - mean) / std

        noise = sp.SparseTensor(
            feats=torch.randn_like(target_feats).to(self.device),
            coords=target_coords.to(self.device),
        )
        return targets, cond, noise

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.slat_flow_model(x, t, cond)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, noise)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        ref_image: Image.Image = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
        init_mesh: trimesh.Trimesh = None,
        coords: torch.Tensor = None,
        normalize_init_mesh: bool = False,
        init_resolution: int = 62,
        init_stride: int = 4
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        if ref_image is not None:
            cond = self.encode_image([image, ref_image])
            neg_cond = torch.zeros_like(cond[0:1])
            sparse_cond = slat_cond = {
                'cond': 0.5 * cond[0:1] + 0.5 * cond[1:2],
                'neg_cond': neg_cond,
            }
        else:
            sparse_cond = slat_cond = self.get_cond([image])

        torch.manual_seed(seed)
        if coords is not None:
            coords = coords
        else:
            coords = self.sample_sparse_structure(sparse_cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def configure_optimizers(self):
        params = list(self.slat_flow_model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TrellisVGGTTo3DPipeline(TrellisImageTo3DPipeline):
    @property
    def device(self) -> torch.device:
        if getattr(self, '_device', None) is not None:
            return self._device
        return super().device

    def get_ss_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        if self.low_vram:
            self.sparse_structure_vggt_cond.to(self.device)
        cond = self.sparse_structure_vggt_cond(aggregated_tokens_list, image_cond)
        if self.low_vram:
            self.sparse_structure_vggt_cond.cpu()
            torch.cuda.empty_cache()
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_slat_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        b, n, _, _ = aggregated_tokens_list[0].shape
        if self.low_vram:
            self.slat_vggt_cond.to(self.device)
        cond = self.slat_vggt_cond(aggregated_tokens_list, image_cond).reshape(b, n, -1, 1024)
        if self.low_vram:
            self.slat_vggt_cond.cpu()
            torch.cuda.empty_cache()
        cond = [c.squeeze(1) for c in cond.split(1, dim=1)]
        neg_cond = [torch.zeros_like(c) for c in cond]
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    @torch.no_grad()
    def vggt_feat(self, image: Union[torch.Tensor, list[Image.Image]]) -> List:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            image = F.interpolate(image, self.default_image_resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.default_image_resolution, self.default_image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.low_vram:
            self.VGGT_model.to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, _ = self.VGGT_model.aggregator(image[None])
        if self.low_vram:
            self.VGGT_model.cpu()
            torch.cuda.empty_cache()

        return aggregated_tokens_list, image

    def run(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        coords: torch.Tensor = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):

        torch.manual_seed(seed)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                aggregated_tokens_list, _ = self.vggt_feat(image)
        b, n, _, _ = aggregated_tokens_list[0].shape
        image_cond = self.encode_image(image).reshape(b, n, -1, 1024)
        
        # if coords is None:
        ss_flow_model = self.models['sparse_structure_flow_model']
        with torch.no_grad():
            ss_cond = self.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, num_samples)
        # Sample structured latent
        ss_sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        reso = ss_flow_model.resolution
        ss_noise = torch.randn(num_samples, ss_flow_model.in_channels, reso, reso, reso).to(self.device)
        if self.low_vram:
            ss_flow_model.to(self.device)
        ss_latent = self.sparse_structure_sampler.sample(
            ss_flow_model,
            ss_noise,
            **ss_cond,
            **ss_sampler_params,
            verbose=True
        ).samples
        if self.low_vram:
            ss_flow_model.cpu()

        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        coords = torch.argwhere(decoder(ss_latent)>0)[:, [0, 2, 3, 4]].int()
        if self.low_vram:
            decoder.cpu()
            torch.cuda.empty_cache()

        # cond = {
        #     'cond': image_cond.reshape(n, -1, 1024),
        #     'neg_cond': torch.zeros_like(image_cond.reshape(n, -1, 1024))[:1],
        # }        
        # slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        # with self.inject_sampler_multi_image('slat_sampler', len(image), slat_steps, mode=mode):
        #     slat = self.sample_slat(cond, coords, slat_sampler_params)
        with torch.no_grad():
            slat_cond = self.get_slat_cond(image_cond, aggregated_tokens_list, num_samples)
        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats), coords, ss_noise

    def run_refine(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        ss_learning_rate: float,
        ss_start_t: float,
        apperance_learning_rate: float,
        apperance_start_t: float,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        ss_noise: torch.Tensor,
        input_points: torch.Tensor,
        ss_refine_type: str = 'No',
        coords: torch.Tensor = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):

        torch.manual_seed(seed)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                aggregated_tokens_list, input_images = self.vggt_feat(image)
        b, n, _, _ = aggregated_tokens_list[0].shape
        image_cond = self.encode_image(image).reshape(b, n, -1, 1024)
        
        if coords is None:
            with torch.no_grad():
                ss_cond = self.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, num_samples)
            ss = torch.zeros(64, 64, 64, dtype=torch.long, device=image_cond.device)
            ss = ss.index_put_((input_points[:,0], input_points[:,1], input_points[:,2]), torch.tensor(1, dtype=ss.dtype, device=ss.device))
            ss = ss[None, None]
            torch.cuda.empty_cache()
            # Sample structured latent
            if ss_refine_type == 'noise':
                coords = self.sample_sparse_structure_opt_noise(ss_cond, ss, ss_learning_rate, num_samples, sparse_structure_sampler_params, ss_noise)
            elif ss_refine_type == 'deltav':
                coords = self.sample_sparse_structure_opt(ss_cond, ss, ss_learning_rate, ss_start_t, num_samples, sparse_structure_sampler_params, ss_noise)
            torch.cuda.empty_cache()

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coords[:,1:].cpu().numpy() / 64 - 0.5)
        # o3d.io.write_point_cloud('outputs/after_coords.ply', pcd)

        # cond = {
        #     'cond': image_cond.reshape(n, -1, 1024),
        #     'neg_cond': torch.zeros_like(image_cond.reshape(n, -1, 1024))[:1],
        # }
        
        # slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')

        # with self.inject_sampler_multi_image('slat_sampler', len(image), slat_steps, mode=mode):
        #     # slat = self.sample_slat(cond, coords, slat_sampler_params)
        #     slat = self.sample_slat_opt(apperance_learning_rate, apperance_start_t, input_images, extrinsics, intrinsics, cond, coords, slat_sampler_params)
        with torch.no_grad():
            slat_cond = self.get_slat_cond(image_cond, aggregated_tokens_list, num_samples)
        slat = self.sample_slat_opt(apperance_learning_rate, apperance_start_t, input_images, extrinsics, intrinsics, slat_cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisVGGTTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisVGGTTo3DPipeline, TrellisVGGTTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisVGGTTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        new_pipeline.VGGT_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        VGGT_model = VGGT.from_pretrained("Stable-X/vggt-object-v0-1")
        new_pipeline.VGGT_model = VGGT_model.to(new_pipeline.device)
        del new_pipeline.VGGT_model.depth_head
        del new_pipeline.VGGT_model.track_head
        # del new_pipeline.VGGT_model.camera_head
        # del new_pipeline.VGGT_model.point_head
        new_pipeline.VGGT_model.eval()

        new_pipeline.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True
        ).to(new_pipeline.device)
        new_pipeline.birefnet_model.eval()
        
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        # for app_refine.py, please uncomment these lines
        os.makedirs("weights", exist_ok=True)
        model, _ = dreamsim(pretrained=True, device=new_pipeline.device, dreamsim_type="dino_vitb16", cache_dir="weights/dreamsim")
        new_pipeline.dreamsim_model = model
        new_pipeline.dreamsim_model.eval()
        new_pipeline.low_vram = False
        new_pipeline._device = None

        return new_pipeline

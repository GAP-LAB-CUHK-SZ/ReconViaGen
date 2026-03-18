import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs, device='cuda'):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).to(device)
        yaw = torch.tensor(float(yaw)).to(device)
        pitch = torch.tensor(float(pitch)).to(device)
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).to(device) * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().to(device), torch.tensor([0, 0, 1]).float().to(device))
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, need_depth=False, opt=False, **kwargs):
    if isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 1024)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    elif isinstance(sample, Gaussian):
        # from ..renderers import GSplatRenderer, GaussianRenderer
        # renderer = GSplatRenderer()
        from ..renderers import GaussianRenderer
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 1024)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, Octree):
        from ..renderers import OctreeRenderer
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite, need_depth=need_depth)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(res['color'].clamp(0, 1) if opt else \
                                 np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'] if opt else res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'] if opt else res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            return_types = kwargs.get('return_types', ["color", "normal", "nocs", "depth", "mask"])
            res = renderer.render(sample, extr, intr, return_types = return_types)
            if 'normal' not in rets: rets['normal'] = []
            if 'color' not in rets: rets['color'] = []
            if 'nocs' not in rets: rets['nocs'] = []
            if 'depth' not in rets: rets['depth'] = []
            if 'mask' not in rets: rets['mask'] = []
            if 'color' in return_types:
                rets['color'].append(res['color'].clamp(0,1) if opt else \
                                    np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            rets['normal'].append(res['normal'].clamp(0,1) if opt else \
                                  np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            rets['nocs'].append(res['nocs'].clamp(0,1) if opt else \
                                np.clip(res['nocs'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            rets['depth'].append(res['depth'] if opt else \
                                 res['depth'].detach().cpu().numpy())
            rets['mask'].append(res['mask'].detach().cpu().numpy().astype(np.uint8))
    return rets

def render_orth_frames(sample, extrinsics, projections, options={}, colors_overwrite=None, verbose=True, **kwargs):
    # Select renderer according to sample type
    if isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 1024)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')

    rets = {}
    for j, extr in tqdm(enumerate(extrinsics), desc='Rendering Orthographic', disable=not verbose):
        res = renderer.render(sample, extr, None, perspective=projections[j], return_types=["normal", "nocs", "depth"])
        if 'normal' not in rets:
            rets['normal'] = []
        if 'color' not in rets:
            rets['color'] = []
        if 'nocs' not in rets:
            rets['nocs'] = []
        if 'depth' not in rets:
            rets['depth'] = []
        rets['normal'].append(np.clip(
            res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
        ).astype(np.uint8))
        rets['nocs'].append(np.clip(
            res['nocs'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
        ).astype(np.uint8))
        rets['depth'].append(res['depth'].detach().cpu().numpy())
    return rets

def get_ortho_projection_matrix(left, right, bottom, top, near, far):
    """
    使用 torch 创建正交投影矩阵, 使用标准的正交投影矩阵公式:
    [ 2/(r-l)      0          0          -(r+l)/(r-l) ]
    [    0      2/(t-b)       0          -(t+b)/(t-b) ]
    [    0         0      -2/(f-n)      -(f+n)/(f-n) ]
    [    0         0          0               1       ]
    """
    projection_matrix = torch.zeros((4, 4), dtype=torch.float32)

    projection_matrix[0, 0] = 2.0 / (right - left)
    projection_matrix[1, 1] = 2.0 / (top - bottom)
    projection_matrix[2, 2] = -2.0 / (far - near)
    projection_matrix[3, 3] = 1.0

    projection_matrix[0, 3] = -(right + left) / (right - left)
    projection_matrix[1, 3] = -(top + bottom) / (top - bottom)
    projection_matrix[2, 3] = (far + near) / (far - near)

    return projection_matrix


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret

def render_ortho_video(sample, resolution=512, ssaa=4, bg_color=(0, 0, 0), num_frames=300, r=2, inverse_direction=False, pitch=-1,  **kwargs):
    if inverse_direction:
        yaws = torch.linspace(3.1415, -3.1415, num_frames)
    else:
        yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    if pitch != -1:
        pitch = pitch * torch.ones(num_frames)
    else:
        pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitchs = pitch.tolist()
    
    ortho_scale = 0.6
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, 40)
    
    projection = get_ortho_projection_matrix(-ortho_scale, ortho_scale, -ortho_scale, ortho_scale, 1e-6, 100).to(extrinsics[0].device)
    projections = [projection] * num_frames
    render_results = render_orth_frames(sample, extrinsics, projections, {'resolution': resolution, 'bg_color': bg_color, 'ssaa': ssaa}, **kwargs)
    render_results.update({'extrinsics': extrinsics, 'intrinsics': None, 'projections': projections})
    return render_results


def render_multiview(sample, resolution=518, ssaa=4, bg_color=(0, 0, 0), num_frames=30, r = 2, fov = 40, random_offset=False, only_color=False, **kwargs):
    if random_offset:
        yaws = []
        pitchs = []
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_frames):
            y, p = sphere_hammersley_sequence(i, num_frames, offset)
            yaws.append(y)
            pitchs.append(p)
    else:
        cams = [sphere_hammersley_sequence(i, num_frames) for i in range(num_frames)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color, 'ssaa': ssaa}, **kwargs)
    return res['color'] if only_color else res, extrinsics, intrinsics

def render_video(sample, resolution=512, ssaa=4, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, 
                 inverse_direction=False, pitch=-1, **kwargs):
    if inverse_direction:
        yaws = torch.linspace(3.1415, -3.1415, num_frames)
        # pitch = 0.25 + 0.5 * torch.sin(torch.linspace(2 * 3.1415, 0, num_frames))
    else:
        yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    if pitch != -1:
        pitch = pitch * torch.ones(num_frames)
    else:
        pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color, 'ssaa': ssaa}, **kwargs)
    res.update({'extrinsics': extrinsics, 'intrinsics': intrinsics})
    return res

def render_condition_images(sample, resolution=512, ssaa=4, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_frames):
        y, p = sphere_hammersley_sequence(i, num_frames, offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    fov = [value_in_radians * 180 / np.pi for value_in_radians in fov]

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, radius, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color, 'ssaa': ssaa}, **kwargs), extrinsics, intrinsics

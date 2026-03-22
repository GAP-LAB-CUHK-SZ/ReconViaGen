#!/root/miniconda3/envs/reconviagen_v05/bin/python3
"""
Render turntable videos for showcase GLB meshes.

For each case and each mesh, renders two ~8-second videos:
  - <stem>_color.mp4  : PBR-shaded color under fixed HDRI lighting (white bg)
  - <stem>_normal.mp4 : surface normal map (black bg)

Output layout:
  show_case/
    case_1/
      renders/
        mvtrellis2_color.mp4
        mvtrellis2_normal.mp4
        reconviagen_color.mp4
        ...
    case_2/
      renders/
        ...

Usage:
    cd /root/jiahao/code/ReconViaGen
    python render_scripts/render_turntable.py
"""

import sys, os

_HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRELLIS2 = os.path.join(_HERE, 'wheels', 'TRELLIS.2')
if _TRELLIS2 not in sys.path:
    sys.path.insert(0, _TRELLIS2)

os.environ['SPCONV_ALGO']              = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['XFORMERS_DISABLED']        = '1'

import torch
import numpy as np
import trimesh
import cv2
import imageio
from tqdm import tqdm

from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial, Texture
from trellis2.renderers import PbrMeshRenderer, EnvMap
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


# ── Config ────────────────────────────────────────────────────────────────────

NUM_FRAMES    = 240       # 240 frames @ 30 fps  ≈  8 s
FPS           = 30
RESOLUTION    = 1024
CAMERA_R      = 2.0
CAMERA_FOV    = 36.0      # degrees
CAMERA_PITCH  = 0.25      # radians  (~14° above horizon, fixed for clean turntable)
SSAA          = 2
PEEL_LAYERS   = 8

HDRI_PATH     = '/root/jiahao/code/TRELLIS.2/assets/hdri/courtyard.exr'
SHOW_CASE_DIR = os.path.join(_HERE, 'show_case')
GLB_NAMES     = ['mvtrellis2.glb', 'reconviagen.glb', 'reconviagen+trellis2.glb']


# ── GLB → MeshWithPbrMaterial ─────────────────────────────────────────────────

def _pil_to_texture(img, channel: int = None) -> Texture:
    """Convert a PIL image to a CUDA Texture.  Optional single-channel slice."""
    arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    if channel is not None:
        arr = arr[:, :, channel:channel + 1]   # keep shape [..., 1]
    return Texture(torch.tensor(arr, dtype=torch.float32, device='cuda'))


def load_glb(path: str) -> MeshWithPbrMaterial:
    """Load a GLB file and return a CUDA MeshWithPbrMaterial."""
    scene = trimesh.load(path)
    mesh  = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    faces    = torch.tensor(mesh.faces,    dtype=torch.int32,   device='cuda')

    # Re-orient: these assets have +Y = top, +Z = front (non-standard GLTF).
    # TRELLIS rendering world is Z-up; camera sits on the +Y side at yaw=0.
    # R_x(+90°): +Y(top) → +Z(world-up), +Z(front) → −Y (faces camera at yaw=π).
    rot = torch.tensor([[1.,  0.,  0.],
                        [0.,  0., -1.],
                        [0.,  1.,  0.]], device='cuda')
    vertices = vertices @ rot.T

    # UV: per-vertex → per-face-corner.
    # GLTF convention: (0,0) = top-left of image.
    # nvdiffrast/OpenGL convention: (0,0) = bottom-left → flip V.
    uv        = mesh.visual.uv.copy()          # [V, 2]  float in [0, 1]
    uv[:, 1]  = 1.0 - uv[:, 1]               # flip V
    uv_coords = torch.tensor(
        uv[mesh.faces], dtype=torch.float32, device='cuda')  # [F, 3, 2]

    # All faces belong to material 0 (single-material GLBs)
    material_ids = torch.zeros(len(mesh.faces), dtype=torch.int32, device='cuda')

    mat = mesh.visual.material

    # ── Base color ────────────────────────────────────────────────────────────
    bc_tex = _pil_to_texture(mat.baseColorTexture) if mat.baseColorTexture is not None else None
    bc_fac_raw = mat.baseColorFactor
    if bc_fac_raw is not None:
        bc_fac = np.array(bc_fac_raw[:3], dtype=np.float32)
        if bc_fac.max() > 1.0:          # trimesh returns 0-255 integers
            bc_fac = bc_fac / 255.0
        bc_fac = bc_fac.tolist()
    else:
        bc_fac = [1.0, 1.0, 1.0]

    # ── Metallic / Roughness ──────────────────────────────────────────────────
    # GLTF packed ORM texture: G = roughness, B = metallic
    mr = mat.metallicRoughnessTexture
    metallic_tex  = _pil_to_texture(mr, channel=2) if mr is not None else None   # B
    roughness_tex = _pil_to_texture(mr, channel=1) if mr is not None else None   # G
    metallic_fac  = float(mat.metallicFactor)  if mat.metallicFactor  is not None else 0.0
    roughness_fac = float(mat.roughnessFactor) if mat.roughnessFactor is not None else 1.0

    pbr = PbrMaterial(
        base_color_texture = bc_tex,
        base_color_factor  = bc_fac,
        metallic_texture   = metallic_tex,
        metallic_factor    = metallic_fac,
        roughness_texture  = roughness_tex,
        roughness_factor   = roughness_fac,
    )

    out = MeshWithPbrMaterial(vertices, faces, material_ids, uv_coords, [pbr])
    # PbrMaterial.__init__ converts base_color_factor via torch.tensor() which
    # always produces a CPU tensor; move it to CUDA manually.
    for m in out.materials:
        m.base_color_factor = m.base_color_factor.cuda()
    return out


# ── Environment map ───────────────────────────────────────────────────────────

def build_envmap() -> EnvMap:
    img = cv2.imread(HDRI_PATH, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(img, dtype=torch.float32, device='cuda'))


# ── Renderer ──────────────────────────────────────────────────────────────────

def build_renderer() -> PbrMeshRenderer:
    r = PbrMeshRenderer()
    r.rendering_options.resolution   = RESOLUTION
    r.rendering_options.near         = 1
    r.rendering_options.far          = 100
    r.rendering_options.ssaa         = SSAA
    r.rendering_options.peel_layers  = PEEL_LAYERS
    return r


# ── Camera path (fixed-pitch turntable) ──────────────────────────────────────

def turntable_cameras():
    # Start at yaw=π so the front of the object (at −Y after re-orientation) faces the camera.
    yaws    = torch.linspace(np.pi, 3 * np.pi, NUM_FRAMES + 1)[:-1].tolist()
    pitches = [CAMERA_PITCH] * NUM_FRAMES
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, CAMERA_R, CAMERA_FOV)


# ── Render loop ───────────────────────────────────────────────────────────────

def render_turntable(mesh, renderer, envmap):
    """
    Returns:
        color_frames  : list of [H, W, 3] uint8  (PBR shaded, white bg)
        normal_frames : list of [H, W, 3] uint8  (surface normals, black bg)
    """
    extrinsics, intrinsics = turntable_cameras()
    color_frames, normal_frames = [], []

    for extr, intr in tqdm(zip(extrinsics, intrinsics), total=NUM_FRAMES, desc='    frames'):
        result = renderer.render(mesh, extr, intr, envmap=envmap)

        def t2np(tensor):
            # tensor: [C, H, W] float32  →  [H, W, C] uint8
            return np.clip(
                tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255,
                0, 255
            ).astype(np.uint8)

        shaded = t2np(result.shaded)   # [H, W, 3]
        normal = t2np(result.normal)   # [H, W, 3]

        alpha = result.alpha.detach().cpu().numpy()   # [H, W]  float in [0,1]
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, np.newaxis]  # [H, W, 1]

        # White background for color
        color  = (shaded.astype(np.float32) * alpha +
                  255.0 * (1.0 - alpha)).astype(np.uint8)

        # Black background for normal
        normal_out = (normal.astype(np.float32) * alpha).astype(np.uint8)

        color_frames.append(color)
        normal_frames.append(normal_out)

    return color_frames, normal_frames


# ── Video I/O ─────────────────────────────────────────────────────────────────

def save_mp4(frames, path, fps=FPS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1,
                     ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
    print(f'    → {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Building renderer …')
    renderer = build_renderer()
    print('Building envmap …')
    envmap = build_envmap()

    cases = sorted(
        d for d in os.listdir(SHOW_CASE_DIR)
        if os.path.isdir(os.path.join(SHOW_CASE_DIR, d))
    )

    for case in cases:
        case_dir = os.path.join(SHOW_CASE_DIR, case)
        out_dir  = os.path.join(case_dir, 'renders')
        os.makedirs(out_dir, exist_ok=True)

        for glb_name in GLB_NAMES:
            glb_path = os.path.join(case_dir, glb_name)
            if not os.path.exists(glb_path):
                print(f'  [skip] {glb_path}')
                continue

            stem = os.path.splitext(glb_name)[0]
            print(f'\n[{case}] {glb_name}')

            print('  Loading mesh …')
            mesh = load_glb(glb_path)

            print('  Rendering …')
            color_frames, normal_frames = render_turntable(mesh, renderer, envmap)

            save_mp4(color_frames,  os.path.join(out_dir, f'{stem}_color.mp4'))
            save_mp4(normal_frames, os.path.join(out_dir, f'{stem}_normal.mp4'))

            del mesh
            torch.cuda.empty_cache()

    print('\nDone.')


if __name__ == '__main__':
    main()

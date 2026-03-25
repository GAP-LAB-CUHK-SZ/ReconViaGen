"""
app_v06.py – ReconViaGen × TRELLIS.2 hybrid demo
=================================================
Stage 1 (SS)     – ReconViaGen VGGT-based sparse structure
Stage 2 (Shape)  – TRELLIS.2 shape_slat  (DINOv3-conditioned)
Stage 3 (Texture)– TRELLIS.2 tex_slat    (DINOv3-conditioned, PBR)
Stage 4 (Decode) – TRELLIS.2 o_voxel GLB export

UI reference: TRELLIS.2 app_mv.py  +  ReconViaGen app_v05.py
"""

import sys, os

# ── Path setup (must precede any trellis2 import) ────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_TRELLIS2 = os.path.join(_HERE, 'wheels', 'TRELLIS.2')
if _TRELLIS2 not in sys.path:
    sys.path.insert(0, _TRELLIS2)
# o_voxel is installed into the conda env (no extra sys.path needed)

os.environ['SPCONV_ALGO']               = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR']  = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']   = 'expandable_segments:True'
os.environ['XFORMERS_DISABLED']         = '1'   # DINOv2's SwiGLUFFNFused falls back to pure PyTorch

# ── Imports ───────────────────────────────────────────────────────────────────
import gradio as gr
from datetime import datetime
import shutil
import uuid
import cv2
import base64, io
from typing import *

import torch
import numpy as np
import imageio
from PIL import Image
import gc

from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR  = os.path.join(_HERE, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
LOW_VRAM = True

MODES = [
    {"name": "Shaded",      "render_key": "shaded"},
    {"name": "Normal",      "render_key": "normal"},
    {"name": "Base color",  "render_key": "base_color"},
    {"name": "Metallic",    "render_key": "metallic"},
    {"name": "Roughness",   "render_key": "roughness"},
]
STEPS        = 8          # number of slider steps (view angles)
DEFAULT_MODE = 0
DEFAULT_STEP = 3


# ── CSS / JS ──────────────────────────────────────────────────────────────────
css = """
.previewer-container {
    position: relative;
    width: 100%;
    height: 520px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.previewer-container .display-row {
    width: 100%; flex-grow: 1; display: flex;
    justify-content: center; align-items: center; min-height: 360px;
}
.previewer-container .previewer-main-image {
    max-width: 100%; max-height: 100%; object-fit: contain; display: none;
}
.previewer-container .previewer-main-image.visible { display: block; }
.previewer-container .mode-row {
    width: 100%; display: flex; gap: 10px;
    justify-content: center; margin-bottom: 10px; flex-wrap: wrap;
}
.previewer-container .mode-btn {
    padding: 4px 12px; border-radius: 14px; border: 2px solid #ddd;
    cursor: pointer; font-size: 13px; background: none;
    opacity: 0.55; transition: all 0.2s;
}
.previewer-container .mode-btn:hover  { opacity: 0.9; }
.previewer-container .mode-btn.active {
    opacity: 1; border-color: var(--color-accent);
    color: var(--color-accent); font-weight: 600;
}
.previewer-container .slider-row {
    width: 100%; display: flex; align-items: center; padding: 0 12px;
}
.previewer-container input[type=range] {
    -webkit-appearance: none; width: 100%;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    height: 6px; background: #ddd; border-radius: 3px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 18px; width: 18px; border-radius: 50%;
    background: var(--color-accent); -webkit-appearance: none;
    margin-top: -6px; box-shadow: 0 2px 4px rgba(0,0,0,.2);
}
"""

head = """
<script>
function refreshView(mode, step) {
    const allImgs = document.querySelectorAll('.previewer-main-image');
    for (let img of allImgs) {
        if (img.classList.contains('visible')) {
            const [_, m, s] = img.id.split('-');
            if (mode === -1) mode = parseInt(m.slice(1));
            if (step === -1) step = parseInt(s.slice(1));
            break;
        }
    }
    allImgs.forEach(img => img.classList.remove('visible'));
    const tgt = document.getElementById('view-m' + mode + '-s' + step);
    if (tgt) tgt.classList.add('visible');
    document.querySelectorAll('.mode-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', idx === mode);
    });
}
function selectMode(mode) { refreshView(mode, -1); }
function onSliderChange(val) { refreshView(-1, parseInt(val)); }
</script>
"""

empty_html = """
<div class="previewer-container">
  <svg style="opacity:.4;height:60px;color:var(--body-text-color)"
    xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="1.2">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <circle cx="8.5" cy="8.5" r="1.5"/>
    <polyline points="21 15 16 10 5 21"/>
  </svg>
</div>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="jpeg", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats':   tex_slat.feats.cpu().numpy(),
        'coords':           shape_slat.coords.cpu().numpy(),
        'res':              res,
    }


def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


# ── Session management ────────────────────────────────────────────────────────

def start_session(req: gr.Request):
    os.makedirs(os.path.join(TMP_DIR, str(req.session_hash)), exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    return [pipeline.preprocess_image(img[0]) for img in images]

def preprocess_videos(video: str) -> List[Image.Image]:
    vid = imageio.get_reader(video, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    frames = []
    for i, frame in enumerate(vid):
        if i % max(int(fps), 1) == 0:
            img = Image.fromarray(frame)
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            frames.append(img)
    vid.close()
    return [pipeline.preprocess_image(f) for f in frames]


# ── 3D generation ─────────────────────────────────────────────────────────────

def image_to_3d(
    image_gallery,
    multi_image_strategy: str,
    seed: int,
    pipeline_type: str,
    ss_source: str,
    # SS params
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    # SLat params
    slat_guidance_strength: float,
    slat_guidance_rescale: float,
    slat_sampling_steps: int,
    slat_rescale_t: float,
    # Shape SLat params
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    # Tex SLat params
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    # Collect images
    if not image_gallery:
        raise gr.Error("Please upload at least one image.")
    images = []
    for item in image_gallery:
        img = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        images.append(img)

    ss_params = {
        "steps":             ss_sampling_steps,
        "cfg_strength":      ss_guidance_strength,
        "cfg_interval":      [0.6, 1.0],
        "guidance_rescale":  ss_guidance_rescale,
        "rescale_t":         ss_rescale_t,
    }
    slat_params = {
        "steps":             slat_sampling_steps,
        "cfg_strength":      slat_guidance_strength,
        "cfg_interval":      [0.6, 1.0],
        "guidance_rescale":  slat_guidance_rescale,
        "rescale_t":         slat_rescale_t,
    }
    shape_slat_params = {
        "steps":             shape_slat_sampling_steps,
        "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale":  shape_slat_guidance_rescale,
        "rescale_t":         shape_slat_rescale_t,
    }
    tex_slat_params = {
        "steps":             tex_slat_sampling_steps,
        "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale":  tex_slat_guidance_rescale,
        "rescale_t":         tex_slat_rescale_t,
    }

    if len(images) == 1:
        out_mesh_list, latents = pipeline.run(
            images, seed=seed,
            ss_sampler_params=ss_params,
            slat_sampler_params=slat_params,
            shape_slat_sampler_params=shape_slat_params,
            tex_slat_sampler_params=tex_slat_params,
            pipeline_type=pipeline_type,
            preprocess_image=True,
            return_latent=True,
            ss_source=ss_source,
        )
    else:
        out_mesh_list, latents = pipeline.run_multi_image(
            images, strategy=multi_image_strategy, seed=seed,
            ss_sampler_params=ss_params,
            slat_sampler_params=slat_params,
            shape_slat_sampler_params=shape_slat_params,
            tex_slat_sampler_params=tex_slat_params,
            pipeline_type=pipeline_type,
            preprocess_image=True,
            return_latent=True,
            ss_source=ss_source,
        )

    mesh = out_mesh_list[0]
    mesh.simplify(16777216)

    render_views = render_utils.render_snapshot(
        mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap
    )
    state = pack_state(latents)
    torch.cuda.empty_cache()

    # ── Build previewer HTML ──────────────────────────────────────────────────
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            uid       = f"view-m{m_idx}-s{s_idx}"
            is_vis    = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_vis else ""
            render_key = mode['render_key']
            frames = render_views.get(render_key)
            if frames is None:
                continue
            img_b64 = image_to_base64(Image.fromarray(frames[s_idx]))
            images_html += f"""
                <img id="{uid}" class="previewer-main-image {vis_class}"
                     src="{img_b64}" loading="eager">
            """

    btns_html = ""
    for idx, mode in enumerate(MODES):
        active = "active" if idx == DEFAULT_MODE else ""
        btns_html += f"""
            <button class="mode-btn {active}" onclick="selectMode({idx})">{mode['name']}</button>
        """

    full_html = f"""
    <div class="previewer-container">
        <div class="display-row">{images_html}</div>
        <div class="mode-row">{btns_html}</div>
        <div class="slider-row">
            <input type="range" min="0" max="{STEPS-1}" value="{DEFAULT_STEP}" step="1"
                   oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    return state, full_html


# ── GLB extraction ────────────────────────────────────────────────────────────

def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    if state is None:
        raise gr.Error("Please generate a 3D model first.")
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    shape_slat, tex_slat, res = unpack_state(state)
    mesh = pipeline.trellis2_pipeline.decode_latent(shape_slat, tex_slat, res)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    now       = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    glb_path  = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


# ── Multi-view examples ───────────────────────────────────────────────────────

def prepare_multi_example() -> List[Image.Image]:
    example_dir = os.path.join(_HERE, "assets", "example_multi_image")
    if not os.path.exists(example_dir):
        return []
    multi_case = list(set([i.split('_')[0] for i in os.listdir(example_dir)]))
    images = []
    for case in multi_case:
        _imgs = []
        for i in range(1, 9):
            p = os.path.join(example_dir, f'{case}_{i}.png')
            if os.path.exists(p):
                img = Image.open(p)
                W, H = img.size
                img = img.resize((int(W / H * 512), 512))
                _imgs.append(np.array(img))
        if _imgs:
            images.append(Image.fromarray(np.concatenate(_imgs, axis=1)))
    return images


def split_image(image) -> List[Image.Image]:
    if image is None:
        return []
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        return []
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    arr   = np.array(image)
    alpha = arr[..., 3]
    cols  = np.any(alpha > 0, axis=0)
    starts = np.where(~cols[:-1] & cols[1:])[0].tolist()
    ends   = np.where(cols[:-1]  & ~cols[1:])[0].tolist()
    return [pipeline.preprocess_image(Image.fromarray(arr[:, s:e+1]))
            for s, e in zip(starts, ends)]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="ReconViaGen × TRELLIS.2",
    delete_cache=(600, 600),
) as demo:

    gr.Markdown("""
    # ReconViaGen × TRELLIS.2 Hybrid
    **Stage 1 – Sparse Structure**: ReconViaGen (VGGT multi-view aware)
    **Stage 2 – Shape SLat**: TRELLIS.2 (DINOv3-conditioned)
    **Stage 3 – Texture SLat**: TRELLIS.2 (DINOv3-conditioned, PBR output)
    """)

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):

            input_video = gr.Video(label="Upload Video", interactive=True, height=220)
            image_prompt = gr.Gallery(
                label="Image Prompts (upload one or more views)",
                columns=3, rows=2, height=250, interactive=True,
                type="pil", file_types=["image"],
            )

            multi_image_strategy = gr.Radio(
                choices=["average_right", "weighted_average", "sequential", "average", "adaptive_guidance_weight", "fixed_guidance_rescale"],
                value="adaptive_guidance_weight",
                label="Multi-image fusion strategy",
                info=(
                    "adaptive_guidance_weight: per-token weight = guidance magnitude ‖v_cond−v_uncond‖, t-adaptive temperature (best) | "
                    "average_right: 1 uncond + N cond calls, CFG applied once on averaged velocity | "
                    "weighted_average: same as average_right but views weighted by deviation from cross-view consensus | "
                    "fixed_guidance_rescale: PoE with per-view independent rescale | "
                    "sequential: cycle images per denoising step (cheapest) | "
                    "average: avg pred_x_prev across images (2N passes, biased rescale)"
                ),
            )
            pipeline_type = gr.Radio(
                choices=["512", "1024", "1536"],
                value="1024",
                label="Output Resolution",
                info="'1024' = higher detail, more VRAM; '512' = faster",
            )
            ss_source = gr.Radio(
                choices=["direct", "mesh", "mvtrellis2"],
                value="mvtrellis2",
                label="Stage 1 Coords Source",
                info=(
                    "direct: ReconViaGen SS diffusion → coords (fast) | "
                    "mesh: ReconViaGen full pipeline → mesh → decimate/fill/voxelize → coords (higher quality) | "
                    "mvtrellis2: TRELLIS.2 SS flow model → coords (multi-image fusion strategy applied)"
                ),
            )

            seed           = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
            decimation_target = gr.Slider(100000, 1000000, label="Decimation Target",
                                          value=500000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size",
                                     value=2048, step=1024)

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("**Stage 1 · Sparse Structure (ReconViaGen)**")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength",
                                                     value=7.5, step=0.1)
                    ss_guidance_rescale  = gr.Slider(0.0, 1.0,  label="Guidance Rescale",
                                                     value=0.7, step=0.01)
                    ss_sampling_steps    = gr.Slider(1, 50, label="Sampling Steps",
                                                     value=30, step=1)
                    ss_rescale_t         = gr.Slider(1.0, 6.0, label="Rescale T",
                                                     value=5.0, step=0.1)

                gr.Markdown("**Stage 2 · Sparse Structure (ReconViaGen)**")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength",
                                                     value=7.5, step=0.1)
                    slat_guidance_rescale  = gr.Slider(0.0, 1.0,  label="Guidance Rescale",
                                                     value=0.5, step=0.01)
                    slat_sampling_steps    = gr.Slider(1, 50, label="Sampling Steps",
                                                     value=12, step=1)
                    slat_rescale_t         = gr.Slider(1.0, 6.0, label="Rescale T",
                                                     value=3.0, step=0.1)

                gr.Markdown("**Stage 3 · Shape SLat (TRELLIS.2)**")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0,
                        label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale  = gr.Slider(0.0, 1.0,
                        label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps    = gr.Slider(1, 50,
                        label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t         = gr.Slider(1.0, 6.0,
                        label="Rescale T", value=3.0, step=0.1)

                gr.Markdown("**Stage 4 · Texture SLat (TRELLIS.2)**")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0,
                        label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale  = gr.Slider(0.0, 1.0,
                        label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps    = gr.Slider(1, 50,
                        label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t         = gr.Slider(1.0, 6.0,
                        label="Rescale T", value=3.0, step=0.1)

        # ── Right panel ───────────────────────────────────────────────────────
        with gr.Column(scale=10):
            preview_output = gr.HTML(empty_html, label="3D Preview", show_label=True)
            extract_btn    = gr.Button("Extract GLB")
            glb_output     = gr.Model3D(label="Extracted GLB", height=480,
                                        display_mode="solid",
                                        clear_color=(0.25, 0.25, 0.25, 1.0))
            download_btn   = gr.DownloadButton(label="Download GLB", interactive=False)

    output_buf = gr.State()

    # Example row
    with gr.Row():
        _dummy_img = gr.Image(visible=False, type="pil", image_mode="RGBA")
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[_dummy_img],
            fn=split_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # ── Event handlers ────────────────────────────────────────────────────────
    demo.load(start_session)
    demo.unload(end_session)

    input_video.upload(preprocess_videos, inputs=[input_video], outputs=[image_prompt])
    input_video.clear(lambda: (None, None), outputs=[input_video, image_prompt])

    # NOTE: removed upload-time preprocessing to avoid double-preprocessing
    # when images are uploaded one at a time. Preprocessing now happens in
    # image_to_3d by passing preprocess_image=True to the pipeline.
    # image_prompt.upload(preprocess_images, inputs=[image_prompt], outputs=[image_prompt])

    generate_btn.click(
        get_seed, inputs=[randomize_seed, seed], outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, multi_image_strategy, seed, pipeline_type, ss_source,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            slat_guidance_strength, slat_guidance_rescale, slat_sampling_steps, slat_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale,
            shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale,
            tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output],
    )

    extract_btn.click(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
    ).then(
        lambda: gr.update(interactive=True), outputs=[download_btn]
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load ReconViaGen pipeline (SS stage)
    print("[1/2] Loading ReconViaGen pipeline (SS stage) …")
    vggt_pipeline = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    vggt_pipeline.cuda()
    vggt_pipeline.VGGT_model.cuda()
    vggt_pipeline.birefnet_model.cuda()
    # Keep slat_flow_model / slat_vggt_cond / slat_decoder_mesh:
    # _run_ss_stage now calls vggt_pipeline.run(formats=["mesh"]) which needs them.
    del vggt_pipeline.models['slat_decoder_gs']   # Gaussian decoder not needed

    if LOW_VRAM:
        # Start vggt models on CPU; _run_ss_stage moves them to GPU when needed.
        vggt_pipeline.VGGT_model.cpu()
        for model in vggt_pipeline.models.values():
            model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    # Load TRELLIS.2 pipeline (shape/tex slat + decode)
    print("[2/2] Loading TRELLIS.2 pipeline (shape/tex slat) …")
    trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    trellis2_pipeline.cuda()
    # del trellis2_pipeline.models['sparse_structure_decoder']
    # del trellis2_pipeline.models['sparse_structure_flow_model']
    if LOW_VRAM:
        trellis2_pipeline.low_vram = True
    gc.collect()
    torch.cuda.empty_cache()

    # Combine into hybrid pipeline
    pipeline = TrellisHybridPipeline(vggt_pipeline, trellis2_pipeline, low_vram=LOW_VRAM)

    # Load HDRI environment maps for PBR rendering
    _HDRI_DIR = os.path.join('/root/jiahao/code/TRELLIS.2', 'assets', 'hdri')
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(os.path.join(_HDRI_DIR, 'courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    demo.launch(css=css, head=head, share=True)

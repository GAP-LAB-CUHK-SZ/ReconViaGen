import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
import uuid
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils



MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image for 3D generation.
    
    This function is called when a user uploads an image or selects an example.
    It applies background removal and other preprocessing steps necessary for
    optimal 3D model generation.

    Args:
        image (Image.Image): The input image from the user

    Returns:
        Image.Image: The preprocessed image ready for 3D generation
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image

def preprocess_videos(video: str) -> List[Tuple[Image.Image, str]]:
    """
    Preprocess the input video for multi-image 3D generation.
    
    This function is called when a user uploads a video.
    It extracts frames from the video and processes each frame to prepare them
    for the multi-image 3D generation pipeline.
    
    Args:
        video (str): The path to the input video file
        
    Returns:
        List[Tuple[Image.Image, str]]: The list of preprocessed images ready for 3D generation
    """
    vid = imageio.get_reader(video, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    images = []
    for i, frame in enumerate(vid):
        if i % max(int(fps * 1), 1) == 0:
            img = Image.fromarray(frame)
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            images.append(img)
    vid.close()
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images for multi-image 3D generation.
    
    This function is called when users upload multiple images in the gallery.
    It processes each image to prepare them for the multi-image 3D generation pipeline.
    
    Args:
        images (List[Tuple[Image.Image, str]]): The input images from the gallery
        
    Returns:
        List[Image.Image]: The preprocessed images ready for 3D generation
    """
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed for generation.
    
    This function is called by the generate button to determine whether to use
    a random seed or the user-specified seed value.
    
    Args:
        randomize_seed (bool): Whether to generate a random seed
        seed (int): The user-specified seed value
        
    Returns:
        int: The seed to use for generation
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def generate_and_extract_glb(
    multiimages: List[Tuple[Image.Image, str]],
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    ss_guidance_rescale: float,
    ss_rescale_t: float,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    slat_guidance_rescale: float,
    slat_rescale_t: float,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    mesh_simplify: float,
    texture_size: int,
    low_vram: bool,
    req: gr.Request,
) -> Tuple[dict, str, str, str]:
    """
    Convert an image to a 3D model and extract GLB file.

    Args:
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
        is_multiimage (bool): Whether is in multi-image mode.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
        str: The path to the extracted GLB file.
        str: The path to the extracted GLB file (for download).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    image_files = [image[0] for image in multiimages]

    # Configure VRAM mode
    pipeline.low_vram = low_vram
    if not low_vram:
        for model in pipeline.models.values():
            model.to(pipeline._device)
        pipeline.VGGT_model.to(pipeline._device)

    # Generate 3D model
    outputs, _, _ = pipeline.run(
        image=image_files,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,

        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": slat_guidance_rescale,
            "rescale_t": slat_rescale_t,
        },
        mode=multiimage_algo,
    )

    # Render video
    # import uuid
    # output_id = str(uuid.uuid4())
    # os.makedirs(f"{TMP_DIR}/{output_id}", exist_ok=True)
    # video_path = f"{TMP_DIR}/{output_id}/preview.mp4"
    # glb_path = f"{TMP_DIR}/{output_id}/mesh.glb"
    
    video_color = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
    del video_color, video_geo
    output_id = str(uuid.uuid4())
    video_path = os.path.join(user_dir, f'{output_id}.mp4')
    imageio.mimsave(video_path, video, fps=15)
    del video

    # Extract GLB
    gs = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]
    torch.cuda.empty_cache()
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, f'{output_id}.glb')
    glb.export(glb_path)
    del glb

    # Pack state for optional Gaussian extraction
    state = pack_state(gs, mesh)
    del outputs

    torch.cuda.empty_cache()
    return state, video_path, glb_path, glb_path


@torch.inference_mode()
def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian splatting file from the generated 3D model.
    
    This function is called when the user clicks "Extract Gaussian" button.
    It converts the 3D model state into a .ply file format containing
    Gaussian splatting data for advanced 3D applications.

    Args:
        state (dict): The state of the generated 3D model containing Gaussian data
        req (gr.Request): Gradio request object for session management

    Returns:
        Tuple[str, str]: Paths to the extracted Gaussian file (for display and download)
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, f'{uuid.uuid4()}.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 9):
            if os.path.exists(f'assets/example_multi_image/{case}_{i}.png'):
                img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
                W, H = img.size
                img = img.resize((int(W / H * 512), 512))
                _images.append(np.array(img))
        if len(_images) > 0:
            images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split a multi-view image into separate view images.
    
    This function is called when users select multi-image examples that contain
    multiple views in a single concatenated image. It automatically splits them
    based on alpha channel boundaries and preprocesses each view.
    
    Args:
        image (Image.Image): A concatenated image containing multiple views
        
    Returns:
        List[Image.Image]: List of individual preprocessed view images
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]

# Create interface
demo = gr.Blocks(
    title="ReconViaGen",
    css="""
        .slider .inner { width: 5px; background: #FFF; }
        .viewport { aspect-ratio: 4/3; }
        .tabs button.selected { font-size: 20px !important; color: crimson !important; }
        h1, h2, h3 { text-align: center; display: block; }
        .md_feedback li { margin-bottom: 0px !important; }
    """
)
with demo:
    gr.Markdown("""
    # 💻 ReconViaGen
    <p align="center">
    <a title="Github" href="https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/GAP-LAB-CUHK-SZ/ReconViaGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Website" href="https://jiahao620.github.io/reconviagen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a title="arXiv" href="https://jiahao620.github.io/reconviagen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    </p>

    ✨This demo is partial. We will release the whole model later. Stay tuned!✨
    """)

    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Input Video or Images", id=0) as multiimage_input_tab:
                    input_video = gr.Video(label="Upload Video", interactive=True, height=300)
                    image_prompt = gr.Image(label="Image Prompt", format="png", visible=False, image_mode="RGBA", type="pil", height=300)
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images.
                                """)

            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=30, step=1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="multidiffusion")
                low_vram = gr.Checkbox(label="Low VRAM Mode (offload models between stages)", value=True)

            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

            generate_btn = gr.Button("Generate & Extract GLB", variant="primary")
            extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("""
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)

            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row() as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    input_video.upload(
        preprocess_videos,
        inputs=[input_video],
        outputs=[multiimage_prompt],
    )
    input_video.clear(
        lambda: tuple([None, None]),
        outputs=[input_video, multiimage_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        generate_and_extract_glb,
        inputs=[multiimage_prompt, seed, ss_guidance_strength, ss_sampling_steps, ss_guidance_rescale, ss_rescale_t,
                slat_guidance_strength, slat_sampling_steps, slat_guidance_rescale, slat_rescale_t, multiimage_algo,
                mesh_simplify, texture_size, low_vram],
        outputs=[output_buf, video_output, model_output, download_glb],
    ).then(
        lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[extract_gs_btn, download_glb],
    )

    video_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[extract_gs_btn, download_glb, download_gs],
    )

    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.update(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[download_glb, download_gs],
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    pipeline._device = torch.device('cuda')
    pipeline.low_vram = True   # default; updated per-request from UI
    pipeline.birefnet_model.cuda()  # small model, keep on GPU permanently
    demo.launch()
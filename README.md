# ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation

<!-- <p align="center">
<a title="Website" href="https://jiahao620.github.io/reconviagen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
  <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
  <img width="1920" height="1080" alt="videoframe_1023" src="https://github.com/user-attachments/assets/08af0af5-5b83-477f-9d4b-a895488befbb" />
</a>
</p> -->

<div align="center">


[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://jiahao620.github.io/reconviagen) 
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2510.23306) 
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/ReconViaGen) 

 </div>

![teaser](assets/Teaser.png)

**Alpha Demo**: https://huggingface.co/spaces/Stable-X/ReconViaGen.
We welcome feedback on failure cases to help improve the model.

**News**: Releasing the training and inference code of ReconViaGen-v0.2. We have optimized the inference process. Reconstructing 16 images using ReconViaGen without refinement (app.py) consumes less than 18GB of VRAM.
Reconstructing 16 images using ReconViaGen (app_fine.py) consumes less than 24GB of VRAM.


**News**: An [unofficial implementation of ReconViaGen](https://github.com/estheryang11/ReconViaGen) is released! Thanks to [estheryang11](https://github.com/estheryang11) a lot!


## Installation
Clone the repo:
```sh
git clone --recursive https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git
cd ReconViaGen
```

Create a new conda environment named reconviagen and install the dependencies (pytorch 2.4.0 with CUDA 12.1):
```sh
. ./setup.sh --new-env --basic --xformers --flash-attn --spconv --mipgaussian --kaolin --nvdiffrast --demo
```

## Local Demo 🤗
Run the script to reconstruct the object without refinement by:
```sh
python app.py
```

Run the script to reconstruct the object with refinement by:
```sh
python app_fine.py
```

## Training

### 0. Data Preparation
The processed dataset can be download [here](https://huggingface.co/datasets/Stable-X/ProObjaverse-300K/tree/main). The dataset is organized as follows:

```
ProObjaverse-300K/
├── renders_random_env/
│   ├── shard-0000/
│   │   ├── {uid}.tar          # per-object archive
│   │   │   ├── {uid}/000.json          # camera metadata (extrinsic 4×4, intrinsic 3×3)
│   │   │   ├── {uid}/000.rgba.webp     # RGBA render, 1024×1024
│   │   │   ├── {uid}/001.json
│   │   │   ├── {uid}/001.rgba.webp
│   │   │   └── ...                     # up to ~80 views per object
│   │   └── ...
│   ├── shard-0001/
│   └── ...
└── lh-slats/
    ├── shard-0000/
    │   ├── {uid}.npz          # structured latent for the object
    │   │   ├── feats:  float32 (N, 8)       # latent features per voxel
    │   │   └── coords: uint8   (N, 3)        # voxel coordinates in [0, 63]
    │   └── ...
    ├── shard-0001/
    └── ...
```

Each `.tar` contains all rendered views for one object. The `uid` is shared between the render tar and the slat npz, and is used to pair them at training time. The `.json` camera file contains all camera pose of rendered views.

### 1. Training DiT of SS Stage.

Run the following code to train the flow model of SS Stage on the ProObjaverse-300K dataset:
```sh
. ./train_ss.sh
```
Noted that we trained the model with 8 A100 GPUs (80GB).

### 2. Training DiT of SLat Stage.

Run the following code to train the flow model of SLat Stage on the ProObjaverse-300K dataset:
```sh
. ./train_slat.sh
```
Noted that we trained the model with 8 A100 GPUs (80GB).

### 3. Try the checkpoint with gradio:

Run the following code to try your trained checkpoints with gradio:

```sh
python app_try.py --ss_ckpt /path_to_your_trained_ss_ckpt --slat_ckpt /path_to_your_trained_slat_ckpt
```


## Citation

```bibtex
@article{chang2025reconviagen,
        title={ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation},
        author={Chang, Jiahao and Ye, Chongjie and Wu, Yushuang and Chen, Yuantao and Zhang, Yidan and Luo, Zhongjin and Li, Chenghong and Zhi, Yihao and Han, Xiaoguang},
        journal={arXiv preprint arXiv:2510.23306},
        year={2025}
}
```

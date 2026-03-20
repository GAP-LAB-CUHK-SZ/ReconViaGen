# ReconViaGen
This repo demonstrates an unofficial pytorch implementation of [ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation](https://jiahao620.github.io/reconviagen/). Thanks for the great work of [Jiahao](https://github.com/Jiahao620)!
![Demo Link](assets/demo.gif)

## Installation
Clone the repo:
```bash
git clone --recursive https://github.com/estheryang11/ReconViaGen.git
cd ReconViaGen
```

Create a conda environment (optional):
```bash
conda create -n reconviagen python=3.10
conda activate reconviagen
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/{your-cuda-version}
pip install spconv-cu{your-cuda-version}==2.3.6 xformers==0.0.27.post2
# other dependencies
pip install -r requirements.txt
```

## Local Demo 🤗
Run by:
```bash
python app_refine.py
```
To improve the accuracy of camera registration, we adjust and optimize the camera pose estimation strategy in ReconViaGen slightly.

# Acknowledgement
The origin paper:
```
@article{chang2025reconviagen,
        title={ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation},
        author={Chang, Jiahao and Ye, Chongjie and Wu, Yushuang and Chen, Yuantao and Zhang, Yidan and Luo, Zhongjin and Li, Chenghong and Zhi, Yihao and Han, Xiaoguang},
        journal={arXiv preprint arXiv:2510.23306},
        year={2025}
}
```

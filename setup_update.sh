pip install zstandard pillow-simd

mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation

mkdir -p /tmp/extensions
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install /tmp/extensions/nvdiffrec --no-build-isolation

mkdir -p /tmp/extensions
git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
pip install /tmp/extensions/CuMesh --no-build-isolation

mkdir -p /tmp/extensions
git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
pip install /tmp/extensions/FlexGEMM --no-build-isolation

mkdir -p /tmp/extensions
cp -r o-voxel /tmp/extensions/o-voxel
pip install /tmp/extensions/o-voxel --no-build-isolation   

pip install gradio==6.0.1
pip uninstall -y gradio-litmodel3d
pip install huggingface_hub==0.36.2
pip install "transformers>=4.57.1"
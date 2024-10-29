conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install -U 'sapien>=3.0.0b1'
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
pip install configargparse tensorboardX tensorboard imageio opencv-python matplotlib tqdm scipy pytorch_msssim jaxtyping plyfile omegaconf tabulate rich kornia
conda install -c iopath iopath
export CUDA_HOME=$CONDA_PREFIX
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
cd msplat & pip install .
cd ..
pip install -r requirements.txt
pip install -e .
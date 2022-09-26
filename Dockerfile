FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN apt-get update && apt install -y htop zsh openslide-tools vim git unzip zip libturbojpeg libvips && apt-get clean

RUN conda install h5py -y

RUN pip install pandas openslide-python opencv-contrib-python kornia gpustat pytorch-lightning==1.5.10 torchmetrics hydra-core albumentations timm==0.4.9 torchstain submitit wandb tqdm tensorboardX matplotlib scipy scikit-image scikit-learn jpeg4py pyvips pyyaml yacs einops psutil simplejson termcolor

#prefetch_generator

#RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd .. && rm -rf apex

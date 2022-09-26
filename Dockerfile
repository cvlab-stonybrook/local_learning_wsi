FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-get update && apt install -y htop zsh openslide-tools vim git unzip zip libturbojpeg libvips && apt-get clean

RUN conda install h5py -y

RUN pip install pandas openslide-python opencv-contrib-python kornia gpustat pytorch-lightning==1.5.10 torchmetrics hydra-core albumentations timm==0.4.9 torchstain submitit wandb tqdm tensorboardX matplotlib scipy scikit-image scikit-learn jpeg4py pyvips pyyaml yacs einops psutil simplejson termcolor
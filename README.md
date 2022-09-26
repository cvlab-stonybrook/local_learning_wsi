# Gigapixel Whole-Slide Images Classification using Locally Supervised Learning
Pytorch implementation for the locally supervised learning framework described in the paper [Gigapixel Whole-Slide Images Classification using Locally Supervised Learning](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_19), [arxiv](https://arxiv.org/abs/2207.08267) (_MICCAI 2022, accepted for oral presentation_).  

<div>
  <img src="imgs/overview.png" width="100%" />
</div>

## TODOs
- [ ] Refine code and provide more explainations.

## Installation
Install [Anaconda/miniconda](https://www.anaconda.com/products/distribution)  
Required packages
```
  $ conda env create --name locallearning anaconda
  $ conda activate locallearning
  $ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  $ pip install wandb pytorch-lightning==1.5.10 torchmetrics albumentations tqdm pandas
``` 
You can also use docker or singularity. We provide the Dockerfile we used in our experiments.


## Contact
If you have any questions or concerns, feel free to report issues or directly contact us (Jingwei Zhang<jingwezhang@cs.stonybrook.edu> or Xin Zhang <x.zhang@u.nus.edu> ). 

## Acknowledgments
Part of our code was borrowed from [InfoPro-Pytorch](https://github.com/blackfeather-wang/InfoPro-Pytorch). Thanks for their outstanding paper.

## Citation
If you use the code or results in your research, please use the following BibTeX entry.  
```
@inproceedings{zhang2022gigapixel,
  title={Gigapixel Whole-Slide Images Classification Using Locally Supervised Learning},
  author={Zhang, Jingwei and Zhang, Xin and Ma, Ke and Gupta, Rajarsi and Saltz, Joel and Vakalopoulou, Maria and Samaras, Dimitris},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={192--201},
  year={2022},
  organization={Springer}
}
```

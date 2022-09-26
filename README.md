# Gigapixel Whole-Slide Images Classification using Locally Supervised Learning
Pytorch implementation for the locally supervised learning framework described in the paper [Gigapixel Whole-Slide Images Classification using Locally Supervised Learning](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_19), [arxiv](https://arxiv.org/abs/2207.08267) and [video](https://youtu.be/_svTenXpjpw) (_MICCAI 2022, accepted for oral presentation_).  

<div>
  <img src="imgs/overview.png" width="100%"  alt="The overview of our framework."/>
</div>

## Installation
Install [Anaconda/miniconda](https://www.anaconda.com/products/distribution).  
Required packages:
```
  $ conda create --name locallearning anaconda=2022.05=py38_0
  $ conda activate locallearning
  $ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  $ pip install opencv-contrib-python pytorch-lightning==1.5.10 torchmetrics albumentations tqdm pandas wandb
``` 
You can also use docker or singularity. We provide the [Dockerfile](Dockerfile) we used in our experiments.
If you do not want to use wandb, you can comment the logger in [main.py](main.py) (line 162 and 167).

## Data organization
Our dataset is organized as csv indicated datasets. All the images should be stored in a single directory and the path of this directory (```dataset_root```) should be passed to ```main.py```. The labels and train/validation/test seperation should be listed in a csv file. This csv file should contain 3 columns: ```slide_id```, ```label``` and ```type```. ```slide_id``` is the filename of input image, including file extension. ```label``` is the integer label of a input WSI. The values in the ```type``` column should be "train", "valid" or "test". A [sample csv file](sample_csv.csv ) is provided. 

## Training
We used ```train.py``` to conduct our training pipeline. 
```
usage: main.py [-h] [--data-mean DATA_MEAN] [--data-std DATA_STD]
               [--num-workers NUM_WORKERS] [--num-classes NUM_CLASSES]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE]
               [--accumulate-grad-batches ACCUMULATE_GRAD_BATCHES] [--K K]
               [--load-weights LOAD_WEIGHTS] [--precision PRECISION] [--lr LR]
               [--lr-factor LR_FACTOR] [--loss-weight LOSS_WEIGHT]
               [--alpha ALPHA] [--decay-multi-epochs DECAY_MULTI_EPOCHS]
               [--weight-decay WEIGHT_DECAY] --output-dir OUTPUT_DIR
               [--project-name PROJECT_NAME] [--gpu-id GPU_ID]
               [--run-name RUN_NAME] [--progressive]
               dataset_root dataset_csv
```
Useful arguments:
```
--output-dir           # The path of directory to store outputs
[--num-classes]        # Number of classes of the input dataset
[--data-mean, --data-std DATA_STD] # the mean and stand devariation of the dataset
[--K]                  # K, the number of divided local modules
[--num_epochs]         # Number of training epochs
[--lr]                 # Initial learning rate
[--lr-factor]          # The multiplication factor on lr of the pretrained networks (eg. 0.5 and --lr 2e-5 means the lr of the pretrained networks is only 1e-5 )
[--loss-weight]        # The weight of each class, used in unbalanced datasets.
[--alpha]              # Hyperparameter alpha
[--decay-multi-epochs] # Epochs to decay lr (e.g. "10,20" means the lr will decay at epch 10 and 20 by a factor of 0.1)
```
Note that ```--batch-size``` should always be 1 as the input images usually have different sizes. We accumulate gradient of several batchs by setting ```--accumulate-grad-batches```, it is equivalent to a larger batch size.

The entire training can be done by, for example:

[//]: # (```)

[//]: # ($ python main.py /data07/shared/jzhang/data/WSI/TCGA-LUADSC/large_img/5x_jpg  ./sample_csv.csv --num-workers 1 --output-dir /data07/shared/jzhang/result/local_learning/test --data-mean 0.6909,0.4654,0.6119 --data-std 0.1786,0.2102,0.1795 --precision 32 --batch-size 1 --epochs 55 --lr 2e-5 --lr-factor 0.5 --loss-weight 0.964218456,1. --accumulate-grad-batches 8 --decay-multi-epochs 25,35,45 --weight-decay 1e-2 --K 4 --alpha 1. --project-name test --run-name test_run --gpu-id 0)

[//]: # (```)
```
$ python main.py path_to_the_image_directory  path_to_the_csv_file --num-workers 4 --output-dir path_to_output_directory --data-mean 0.6909,0.4654,0.6119 --data-std 0.1786,0.2102,0.1795 --precision 32 --batch-size 1 --epochs 55 --lr 2e-5 --lr-factor 0.5 --loss-weight 0.964218456,1. --accumulate-grad-batches 8 --decay-multi-epochs 25,35,45 --weight-decay 1e-2 --K 4 --alpha 1. --project-name test --run-name test_run --gpu-id 0
```

## Contact
If you have any questions or concerns, feel free to report issues or directly contact us (Jingwei Zhang <jingwezhang@cs.stonybrook.edu> or Xin Zhang <x.zhang@u.nus.edu>). 

## Acknowledgments
Part of our code was borrowed from [InfoPro-Pytorch](https://github.com/blackfeather-wang/InfoPro-Pytorch). Thanks for their outstanding paper.
Our framework used [Pytorch Lightning](https://github.com/Lightning-AI/lightning). Thanks for this simple and efficient framework which faciliated our developments.

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

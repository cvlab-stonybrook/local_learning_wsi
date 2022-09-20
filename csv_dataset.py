import os
import cv2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
# import jpeg4py as jpeg


def read_rgb_img(img_path):
    # return Image.open(img_path).convert('RGB')
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


# def read_rgb_jpg(img_path):
#     return jpeg.JPEG(img_path).decode()


class CsvDataset(Dataset):

    def __init__(self, dataset_path, dataset_csv, data_type, transforms=transforms.ToTensor(), **kwargs):
        self.root_path = dataset_path
        self.dataset_csv_path = dataset_csv
        if data_type not in ['train', 'validation', 'test']:
            raise Exception("Not supported dataset type. It should be train, validation or test")
        self.data_type = data_type
        if data_type == 'test':
            self.val_fold_id = -1

        # read csv and store label
        self.label_df = self.read_dataset_csv()
        self.label = self.label_df['label'].values
        self.image_id = self.label_df['slide_id'].values

        self.transforms = transforms

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        img_id, label = self.image_id[i], self.label[i]
        file_name = img_id
        full_path = os.path.join(self.root_path, file_name)

        # if self.data_extension == "jpg":
        #     img = read_rgb_jpg(full_path)
        # else:
        img = read_rgb_img(full_path)

        if self.transforms is not None:
            img = self.transforms(image=img)
            img = img["image"]
            if isinstance(img, np.ndarray):
                img = transforms.ToTensor()(img)

        return img, label

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type == 'validation':
            df = df[int(df['type']) == 1]
        elif self.data_type == 'test':
            df = df[int(df['type']) == 2]
        else:
            df = df[int(df['type']) == 0]
        return df

    def get_weights_of_class(self):
        unique, counts = np.unique(self.label, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=np.float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))


class CsvDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, dataset_csv, batch_size, cus_transforms=transforms.ToTensor(), num_workers=2):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_csv = dataset_csv

        # self.cus_transforms = cus_transforms
        if cus_transforms is None:
            self.transforms_train, self.transforms_eval = None, None
        elif isinstance(cus_transforms, (list, tuple)):
            self.transforms_train = cus_transforms[0]
            self.transforms_eval = cus_transforms[1]
        else:
            self.transforms_train, self.transforms_eval = cus_transforms, cus_transforms

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage=None):
        if self.dataset_train is None:
            self.dataset_train = CsvDataset(self.dataset_path, self.dataset_csv, "train", val_fold_id=self.val_fold_id,
                                            data_ext=self.data_ext, transforms=self.transforms_train,
                                            classes_names=[self.CLASSES, self.CLASS_NAMES])

            self.dataset_val = CsvDataset(self.dataset_path, self.dataset_csv, "train",
                                          val_fold_id=self.val_fold_id,
                                          data_ext=self.data_ext, transforms=self.transforms_eval,
                                          classes_names=[self.CLASSES, self.CLASS_NAMES])
            self.dataset_test = CsvDataset(self.dataset_path, self.dataset_csv, "train",
                                           val_fold_id=self.val_fold_id,
                                           data_ext=self.data_ext, transforms=self.transforms_eval,
                                           classes_names=[self.CLASSES, self.CLASS_NAMES])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False, persistent_workers=True)

    def get_train_dataloader(self, shuffle=True):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        return (DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False, persistent_workers=True),
                DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                           drop_last=False, pin_memory=False, persistent_workers=True), )

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)
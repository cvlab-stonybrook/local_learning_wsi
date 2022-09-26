import json
import math
import os
import shutil
import sys
import random

import cv2
from albumentations import DualTransform
import albumentations.augmentations.crops.functional as F


def save_parameters(args):
    folder_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(folder_path, "parameters.json"), "w") as f:
        json.dump(
            {n: str(args_dict[n]) for n in args_dict},
            f,
            indent=4
        )


def save_model_structure(args, model):
    folder_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "model.txt"), "w") as f:
        f.write(str(model))

    # save the running source file
    source_file_path = sys.argv[0]
    source_file_name = os.path.split(source_file_path)[-1]
    try:
        shutil.copy2(source_file_path, os.path.join(folder_path, source_file_name))
    except:
        print("ERROR! Could not copy source file.")


class _BaseRandomCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(_BaseRandomCrop, self).__init__(always_apply, p)
        self.interpolation = interpolation

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return crop

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        return keypoint


class RandomCrop(_BaseRandomCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):

        super(RandomCrop, self).__init__(
            interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]
        scale = self.scale

        for _attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "scale", "ratio", "interpolation"


class RandomCropEdge(_BaseRandomCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.5, 1.0),
        scale_for_small=(0.9, 1.0),
        small_length=2000,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):

        super(RandomCropEdge, self).__init__(
            interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.scale = scale
        self.scale_for_small = scale_for_small
        self.small_length = small_length

    def get_params_dependent_on_targets(self, params):
        img = params["image"]

        hw = [0, 0]
        for i in range(2):
            #####: Major modification here
            scale = self.scale if img.shape[i] >= self.small_length else self.scale_for_small
            hw[i] = random.uniform(*scale) * img.shape[i]
            hw[i] = int(round(hw[i]))
        h, w = hw

        if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
            i = random.randint(0, img.shape[0] - h)
            j = random.randint(0, img.shape[1] - w)
            return {
                "crop_height": h,
                "crop_width": w,
                "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
            }

        # else, do not crop
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": 0.,
            "w_start": 0.,
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "scale", "scale_for_small", "small_length", "interpolation"
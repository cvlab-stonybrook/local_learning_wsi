import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_classifier import BaseClassifier


class InfoProDecoder(BaseClassifier):
    def __init__(self, inplanes, interpolate_mode='bilinear', middle_planes=32, outplanes=64, loss=None):
        super(InfoProDecoder, self).__init__(require_x=True, return_y=False, require_label=False, return_loss=True)

        # self.image_size = image_size

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        # self.loss = nn.MSELoss() #nn.BCELoss()
        self.loss = nn.MSELoss() if loss is None else loss

        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, int(middle_planes), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(middle_planes)),
            nn.ReLU(),
            nn.Conv2d(int(middle_planes), outplanes, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
        )


    def forward(self,  features, x=None, label=None):
        image_ori = x
        # self.mean = self.mean.to(features.device)
        # self.std = self.std.to(features.device)
        if self.interpolate_mode == 'bilinear':
            features_out = F.interpolate(features, size=image_ori.shape[-2:],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features_out = F.interpolate(features, size=image_ori.shape[-2:],
                                     mode='nearest')
        else:
            raise NotImplementedError
        image_rec = self.decoder(features_out)
        # print(image_rec.max(), image_rec.min(), image_ori.max(), image_ori.min())
        return self.loss(image_rec, image_ori)


class RandomInfoProDecoder(InfoProDecoder):
    def __init__(self, inplanes, up_scale, patch_size, num_patches, interpolate_mode='bilinear', middle_planes=32,
                 outplanes=64, loss=None):
        super(RandomInfoProDecoder, self).__init__(inplanes, interpolate_mode, middle_planes, outplanes, loss=loss)

        self.up_scale = up_scale
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, features, x=None, label=None):
        features_large, image_ori_large = features, x

        sampling_space_large = features_large.shape[-2] - self.patch_size, features_large.shape[-1] - self.patch_size

        sampling_space_ori = (image_ori_large.shape[-2] - self.patch_size * self.up_scale) // self.up_scale, \
                             (image_ori_large.shape[-1] - self.patch_size * self.up_scale) // self.up_scale

        sampling_space = min(sampling_space_large[0], sampling_space_ori[0]), \
                min(sampling_space_large[1], sampling_space_ori[1])

        sampling_pos = torch.randint(0, sampling_space[0] * sampling_space[1],
                                     [features_large.shape[0], self.num_patches], device=features.device)
        # sampling_pos = torch.unre
        sampling_pos = [sampling_pos // sampling_space[1], sampling_pos % sampling_space[1]]

        features = []
        image_ori = []
        for bi in range(features_large.shape[0]):
            for si in range(self.num_patches):
                patch_sz = self.patch_size
                loc = sampling_pos[0][bi, si], sampling_pos[1][bi, si]
                features.append(features_large[bi, :, loc[0]:loc[0] + patch_sz, loc[1]:loc[1] + patch_sz])
                loc = loc[0] * self.up_scale, loc[1] * self.up_scale
                patch_sz = self.patch_size * self.up_scale
                image_ori.append(image_ori_large[bi, :, loc[0]:loc[0] + patch_sz, loc[1]:loc[1] + patch_sz])

        features = torch.stack(features, dim=0)
        image_ori = torch.stack(image_ori, dim=0)

        if self.interpolate_mode == 'bilinear':
            features_out = F.interpolate(features, scale_factor=self.up_scale,
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features_out = F.interpolate(features, scale_factor=self.up_scale,
                                     mode='nearest')
        else:
            raise NotImplementedError
        image_rec = self.decoder(features_out)
        return self.loss(image_rec, image_ori)
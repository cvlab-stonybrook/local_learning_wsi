import torch
from torch import nn
import torch.nn.functional as F

from .base_classifier import BaseClassifier


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class AuxClassifier(BaseClassifier):

    def __init__(self, inplanes, net_config='1c2f', loss_mode='cross_entropy', class_num=2, widen=1, feature_dim=128,
                 pooling=nn.AdaptiveAvgPool2d((1, 1)), kernel_size=3, stride=1, padding=1, bias=False,
                 label_smoothing=0.1, loss_weight=None):
        super(AuxClassifier, self).__init__(require_x=False, return_y=True, require_label=True, return_loss=True)

        # assert inplanes in [16, 32, 64, 128, 256, 512]
        assert net_config in ['0c1f', '0c2f', '1c1f', '1c2f', '1c3f', '2c2f']
        assert loss_mode in ['contrast', 'cross_entropy', 'label_smooth_cross_entropy', None]

        self.loss_mode = loss_mode
        self.feature_dim = feature_dim

        # if loss_mode == 'contrast':
        #     self.criterion = SupConLoss()
        #     self.fc_out_channels = feature_dim
        if loss_mode == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
            self.fc_out_channels = class_num
        elif loss_mode == 'label_smooth_cross_entropy':
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.fc_out_channels = class_num
        elif loss_mode == None:
            self.criterion = None
            self.fc_out_channels = class_num
        else:
            raise NotImplementedError

        if net_config == '0c1f':  # Greedy Supervised Learning (Greedy SL)
            self.head = nn.Sequential(
                pooling,
                nn.Flatten(),
                nn.Linear(inplanes, self.fc_out_channels),
            )

        if net_config == '0c2f':
            self.head = nn.Sequential(
                pooling,
                nn.Flatten(),
                nn.Linear(inplanes, int(feature_dim * widen)),
                nn.ReLU(inplace=True),
                nn.Linear(int(feature_dim * widen), self.fc_out_channels)
            )
        if net_config == '1c1f':
            if inplanes < 64:
                out_cnn_dim = int(inplanes * 2 * widen)
            elif inplanes >= 64:
                out_cnn_dim = int(inplanes * widen)

            self.head = nn.Sequential(
                nn.Conv2d(inplanes, out_cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.GroupNorm(out_cnn_dim, out_cnn_dim),
                nn.ReLU(),
                pooling,
                nn.Flatten(),
                nn.Linear(int(out_cnn_dim), self.fc_out_channels),
            )

        if net_config == '1c2f':
            if inplanes < 64:
                out_cnn_dim = int(inplanes * 2 * widen)
            elif inplanes >= 64:
                out_cnn_dim = int(inplanes * widen)

            self.head = nn.Sequential(
                nn.Conv2d(inplanes, out_cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.GroupNorm(out_cnn_dim, out_cnn_dim),
                nn.ReLU(),
                pooling,
                nn.Flatten(),
                nn.Linear(out_cnn_dim, int(feature_dim * widen)),
                nn.ReLU(inplace=True),
                nn.Linear(int(feature_dim * widen), self.fc_out_channels)
            )

        if net_config == '1c3f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(32 * widen), int(32 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '2c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(32 * widen), int(32 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(32 * widen), int(32 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(32 * widen), int(32 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.GroupNorm(int(64 * widen), int(64 * widen)),
                    nn.ReLU(),
                    pooling,
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

    def forward(self, features, x=None, label=None):
        features = self.head(features)
        # if self.loss_mode == 'contrast':
        #     assert features.size(1) == self.feature_dim
        #     features = F.normalize(features, dim=1)
        #     features = features.unsqueeze(1)
        #     loss = self.criterion(features, label, temperature=0.07)
        if self.loss_mode == 'cross_entropy' or self.loss_mode == 'label_smooth_cross_entropy':
            loss = self.criterion(features, label)
        elif self.loss_mode is None:
            loss = features
        else:
            raise NotImplementedError

        return loss, features


class AuxClassifierContainer(BaseClassifier):

    def __init__(self, network, aux_classifier):
        super(AuxClassifierContainer, self).__init__(require_x=False, return_y=True, require_label=True, return_loss=True)
        self.network = network
        self.aux_classifier = aux_classifier

    def forward(self, features, x=None, label=None):
        features = self.network(features)
        return self.aux_classifier(features, x, label)
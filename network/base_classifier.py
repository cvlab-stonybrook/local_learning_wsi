from typing import Optional, Iterable

import numpy
import torch
from torch import nn


class BaseClassifier(nn.Module):
    def __init__(self, require_x, return_y, require_label=False, return_loss=True):
        super(BaseClassifier, self).__init__()
        self.clf_loss_dict = {
            "require_x": require_x,
            "require_label": require_label,
            "return_y": return_y,
            "return_loss": return_loss,
        }

    def check_clf(self, item):
        return self.clf_loss_dict[item]


class ClassifierLossModuleList(nn.ModuleList):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None, alphas: Optional = None) -> None:
        super(ClassifierLossModuleList, self).__init__(modules)

        self.clf_loss_dict = {
            "require_x": False,
            "require_label": False,
            "return_y": False,
            "return_loss": True,  # always return loss
        }

        if alphas is None:
            self.alpha = torch.ones((len(self),), requires_grad=False)
        else:
            assert len(self) == len(alphas)
            self.alpha = torch.tensor(alphas, requires_grad=False)

        for clf_loss in self:
            if isinstance(clf_loss, BaseClassifier):
                for it in clf_loss.clf_loss_dict:
                    if clf_loss.clf_loss_dict[it]:
                        self.clf_loss_dict[it] = True
            else:
                self.clf_loss_dict["return_loss"] = True

    def forward(self, features, x=None, label=None):
        loss, y = None, None
        for clf_loss, alpha_i in zip(self, self.alpha):
            if isinstance(clf_loss, BaseClassifier):
                ret = clf_loss(features, x, label)
                if clf_loss.check_clf("return_y") and clf_loss.check_clf("return_loss"):
                    l = ret[0]
                    loss = l * alpha_i if loss is None else loss + l * alpha_i
                    y = ret[1]
            else:
                l = clf_loss(features)
                loss = l * alpha_i if loss is None else loss + l * alpha_i
        if y is not None:
            return loss, y
        else:
            return loss
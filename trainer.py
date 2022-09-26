import gc
import statistics
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from torch import nn
from torch.optim import lr_scheduler

from network.base_classifier import ClassifierLossModuleList


class LocalModule(pl.LightningModule):
    def __init__(self, backbone, loss_networks, metrics_fn, args, num_classes, valid_as_train=True):
        super(LocalModule, self).__init__()
        # disable auto optim
        self.automatic_optimization = False

        self.args = args
        self.model = backbone

        # adding a dummy loss network
        loss_networks.insert(0, ClassifierLossModuleList())
        self.loss_nets = nn.ModuleList(loss_networks)

        self.K = self.model.K

        self.lr = args.lr
        self.accumulate_grad_batches = args.accumulate_grad_batches

        self.train_metrics, self.eval_metrics, self.test_metrics = metrics_fn

        self.acc_metrics = torchmetrics.Accuracy(num_classes=num_classes)

        eval_test_prefix = "test_" + self.eval_metrics.prefix if self.eval_metrics.prefix is not None else None
        eval_test_postfix = self.eval_metrics.postfix + "_test" if self.eval_metrics.postfix is not None else None
        self.eval_test_metrics = self.eval_metrics.clone(prefix=eval_test_prefix, postfix=eval_test_postfix)

        self.valid_as_train=valid_as_train

        self.save_hyperparameters("args")


    def forward(self, x, label=None, ki=-1):
        # in lightning, forward defines the prediction/inference actions
        if ki == 0:
            return self.model(x, ki)

        loss_fn = self.loss_nets[ki] if ki > 0 else self.loss_nets[self.K]
        label = label if loss_fn.clf_loss_dict["require_label"] else None

        features = self.model(x, ki=ki)

        if not loss_fn.clf_loss_dict["require_x"]:
            x = None

        loss = self.loss_nets[ki](features, x=x, label=label)
        y = None
        if isinstance(loss, (list, tuple)):
            loss, y = loss
        return features, y, loss

    def training_step(self, batch, batch_idx):
        img, label = batch

        retry_time = 0
        while retry_time < 2:
            try:
                if retry_time >= 1:
                    print("Halfing image size")
                    with torch.no_grad():
                        img_size = img.shape[-2:]
                        img_size = img_size[0] // 6, img_size[1] // 6
                        img = img[:, :, img_size[0]: -img_size[0], img_size[1]: -img_size[1]]
                ft = self(img, label=None, ki=0)

                for cur_k in range(1, self.K + 1):
                    ft, y_k, loss_k = self(ft, label, ki=cur_k)
                    y_prob_k = F.softmax(y_k, dim=1)

                    self.manual_backward(loss_k / self.accumulate_grad_batches)
                    ft = ft.detach()

                    self.log("loss/train_%d" % cur_k, loss_k, on_step=True, on_epoch=True, sync_dist=True)
                    self.log("acc/train_%d" % cur_k, self.acc_metrics(y_prob_k, label), on_step=False, on_epoch=True)

                    # gc.collect()
                    # torch.cuda.empty_cache()

                opt = self.optimizers()
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    opt.step()
                    opt.zero_grad()

                loss = loss_k
                y_prob = y_prob_k
                break
            except RuntimeError as e:
                retry_time += 1
                print('Runtime Error {}\nRun Again......{}/{}'.format(e, retry_time, 2))
                # gc.collect()
                # torch.cuda.empty_cache()
                if retry_time >= 2:
                    print('Give up!')
                    return {"loss": torch.zeros(1, device=label.device).mean(),
                            "y_prob_batch": None, "label_batch": None}
                    # return None


        return {"loss": loss.detach(), "y_prob_batch": y_prob.detach(), "label_batch": label.detach()}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        img, label = batch

        # ft, y, loss = self(img, label, ki=-1)
        if self.valid_as_train:
            ft = self(img, label=None, ki=0)
            with torch.no_grad():
                for cur_k in range(1, self.K + 1):
                    ft, y_k, loss_k = self(ft, label, ki=cur_k)
                    y_prob_k = F.softmax(y_k, dim=1)
                    self.log("loss/part_%d" % cur_k, loss_k, on_step=True, on_epoch=True, sync_dist=True)
                    self.log("acc/part_%d" % cur_k, self.acc_metrics(y_prob_k, label), on_step=False, on_epoch=True)
        else:
            ft, y_k, loss_k = self(img, label, ki=-1)

        # y_prob = F.softmax(y, dim=1)
        y_prob = y_k
        loss = loss_k

        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("acc", self.acc_metrics(y_prob, label), on_step=False, on_epoch=True)
        return {"loss": loss.detach(), "y_prob_batch": y_prob.detach(), "label_batch": label.detach()}

    def test_step(self, batch, batch_idx):
        img, label = batch

        ft, y, loss = self(img, label, ki=-1)
        y_prob = F.softmax(y, dim=1)

        self.log("loss/test", loss, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "y_prob_batch": y_prob.detach(), "label_batch": label.detach()}

    def training_epoch_end(self, outs):
        y_prob = torch.cat([o["y_prob_batch"] for o in outs if o is not None and o["y_prob_batch"] is not None], dim=0)
        label = torch.cat([o["label_batch"] for o in outs if o is not None and o["label_batch"] is not None], dim=0)
        metrics = self.train_metrics(y_prob, label)
        # metrics['step'] = self.current_epoch
        self.logger.log_metrics(metrics, step=self.current_epoch)

        sch = self.lr_schedulers()
        sch.step()

    def validation_epoch_end(self, outs_dl):
        for outs, metrics_fn in zip(outs_dl, [self.eval_metrics, self.eval_test_metrics]):
            y_prob = torch.cat([o["y_prob_batch"] for o in outs if o is not None], dim=0)
            label = torch.cat([o["label_batch"] for o in outs if o is not None], dim=0)
            metrics = metrics_fn(y_prob, label)
            if not self.trainer.sanity_checking:
                self.logger.log_metrics(metrics, step=self.current_epoch)

    def test_epoch_end(self, outs):
        y_prob = torch.cat([o["y_prob_batch"] for o in outs], dim=0)
        label = torch.cat([o["label_batch"] for o in outs], dim=0)
        metrics = self.test_metrics(y_prob, label)
        # metrics['step'] = self.current_epoch
        self.logger.log_metrics(metrics)

        print("Testing metrics:")
        print(metrics)

        return metrics

    def configure_optimizers(self):
        if self.args.weight_decay is None:
            self.args.weight_decay = 1e-2
        cus_optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.lr * self.args.lr_factor},
                {"params": self.loss_nets.parameters(), }
            ],
            # self.parameters(),
            lr=self.lr, weight_decay=self.args.weight_decay)

        cus_sch = lr_scheduler.MultiStepLR(cus_optimizer, self.args.decay_multi_epochs, last_epoch=-1, verbose=True)
        return {
            "optimizer": cus_optimizer,
            "lr_scheduler": cus_sch
        }
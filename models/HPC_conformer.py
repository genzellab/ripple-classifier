#%%
import sklearn.metrics as metrics
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser, Namespace


import wandb
import pandas as pd
from torchmetrics.functional import f1_score
import torchmetrics
import torchaudio

class cnn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.0):
        super(cnn_block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class HPC_Conformer(pl.LightningModule):
    def __init__(self, hparams):
        super(HPC_Conformer, self).__init__()
        # if not isinstance(hparams, Namespace):
        #     hparams = dotdict(hparams)

        self.save_hyperparameters(hparams)
        self.num_classes = self.hparams.num_classes
        print('model',hparams)
        self.net = torchaudio.models.Conformer(
            input_dim=hparams.hpc_wavelet_scales_num,
            num_heads=self.hparams.hpc_num_heads,  # number of heads in multiheadattention models
            ffn_dim=self.hparams.hpc_ffn_dim,  # dimension of feedforward network model
            num_layers=self.hparams.hpc_num_layers,  # number of decoder layers
            depthwise_conv_kernel_size=self.hparams.hpc_depthwise_conv_kernel_size,
            use_group_norm=self.hparams.hpc_use_group_norm,
            dropout=self.hparams.hpc_dropout,
            convolution_first=self.hparams.hpc_convolution_first,

        )
        if self.hparams.hpc_get_emb:
            self.fc = nn.Linear(8, self.hparams.hpc_emb_dim)
        else:
            self.fc = nn.Linear(hparams.hpc_wavelet_scales_num, self.num_classes)
        self.init_weights()
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        lengths = torch.full((x.shape[0],), x.shape[2], device=x.device)
        x = x.permute(0, 2, 1)
        x = self.net(x, lengths)[0]
        x = x[:, 0, :]
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return y_hat, y

    def training_step_end(self, batch_parts_outputs):
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss = F.cross_entropy(y_hat, y)
        y = y.detach()
        y_hat = y_hat.detach()
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        acc = metrics.accuracy_score(y.cpu(), y_pred.cpu())

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat, y

    def validation_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        # do softmax here
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        # print(y_pred.shape,y.shape)
        acc = torchmetrics.functional.accuracy(y_pred, y)
        prec_micro = torchmetrics.functional.precision(
            y_pred, y, average='micro', num_classes=self.num_classes)
        prec_macro = torchmetrics.functional.precision(
            y_pred, y, average='macro', num_classes=self.num_classes)
        prec_weighted = torchmetrics.functional.precision(
            y_pred, y, average='weighted', num_classes=self.num_classes)
        #f1_val = f1_score(y_pred, y, num_classes=self.num_classes)
        f1_val = f1_score(y_pred,
                          y,
                          num_classes=self.num_classes,
                          average='macro')
        return {
            "val_loss": loss_val,
            "val_f1": f1_val,
            "val_acc": acc,
            "val_prec_micro": prec_micro,
            "val_prec_macro": prec_macro,
            "val_prec_weighted": prec_weighted,
        }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        tqdm_dict = {}

        for metric_name in [
                "val_loss",
                "val_f1",
                "val_acc",
                "val_prec_micro",
                "val_prec_macro",
                "val_prec_weighted",
        ]:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]
                # reduce manually when using dp
                # if self.trainer.use_dp or self.trainer.use_ddp2:
                #     metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat, y

    def test_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        # do softmax here
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]

        acc = torchmetrics.functional.accuracy(y_pred, y)

        prec_micro = torchmetrics.functional.precision(
            y_pred, y, average='micro', num_classes=self.num_classes)
        prec_macro = torchmetrics.functional.precision(
            y_pred, y, average='macro', num_classes=self.num_classes)
        prec_weighted = torchmetrics.functional.precision(
            y_pred, y, average='weighted', num_classes=self.num_classes)

        rec_micro = torchmetrics.functional.recall(
            y_pred, y, average="micro", num_classes=self.num_classes)
        rec_macro = torchmetrics.functional.recall(
            y_pred, y, average="macro", num_classes=self.num_classes)
        rec_weighted = torchmetrics.functional.recall(
            y_pred, y, average="weighted", num_classes=self.num_classes)

        f1_micro = f1_score(y_pred,
                            y,
                            num_classes=self.num_classes,
                            average="micro")
        f1_macro = f1_score(y_pred,
                            y,
                            num_classes=self.num_classes,
                            average="macro")
        f1_weighted = f1_score(y_pred,
                               y,
                               num_classes=self.num_classes,
                               average="weighted")

        return {
            "test_loss": loss_val,
            "test_acc": acc,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "test_prec_micro": prec_micro,
            "test_prec_macro": prec_macro,
            "test_prec_weighted": prec_weighted,
            "test_rec_micro": rec_micro,
            "test_rec_macro": rec_macro,
            "test_rec_weighted": rec_weighted,
            "res": (y_pred.cpu(), y.cpu()),
        }

    def test_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in [
                "test_loss",
                "test_acc",
                "test_f1_micro",
                "test_f1_macro",
                "test_f1_weighted",
                "test_prec_micro",
                "test_prec_macro",
                "test_prec_weighted",
                "test_rec_micro",
                "test_rec_macro",
                "test_rec_weighted",
                "res",
        ]:
            if metric_name == "res":
                y_pred = torch.zeros((1))
                y = torch.zeros((1))

                for output in outputs:
                    n_ypred, n_y = output[metric_name]
                    y_pred = torch.cat((y_pred, n_ypred))
                    y = torch.cat((y, n_y))
                conf_matrix = torchmetrics.functional.confusion_matrix(y_pred.type(torch.ByteTensor), y.type(torch.ByteTensor),
                                                                       num_classes=self.num_classes).cpu().numpy()
                df = pd.DataFrame(data=y_pred, columns=["pred"])
                df["true"] = y

                report = metrics.classification_report(
                    y.type(torch.ByteTensor),
                    y_pred.type(torch.ByteTensor),
                    output_dict=True,
                )
                print(metrics.classification_report(
                    y.type(torch.ByteTensor),
                    y_pred.type(torch.ByteTensor),
                ))
            else:
                metric_total = 0
                for output in outputs:
                    metric_value = output[metric_name]

                    # # reduce manually when using dp
                    # if self.trainer.use_dp or self.trainer.use_ddp2:
                    #     metric_value = metric_value.mean()

                    metric_total += metric_value

                tqdm_dict[metric_name] = metric_total / len(outputs)
        unique_label = list(range(self.num_classes))
        print(unique_label)
        print(conf_matrix)
        # print(df_pred)
        cmtx = pd.DataFrame(
            conf_matrix,
            index=["true:{:}".format(x) for x in unique_label],
            columns=["pred:{:}".format(x) for x in unique_label],
        )
        report_log = pd.DataFrame(report)
        print(report)

        dict_a = {
            "confusion_matrix":
            wandb.plots.HeatMap(unique_label,
                                unique_label,
                                cmtx.values,
                                show_text=True),
            "classification_report":
            wandb.Table(dataframe=report_log),
            "outputs":
            wandb.Table(dataframe=df),
        }
        self.logger.experiment.log(dict_a)

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            self.hparams.hpc_learning_rate,
            weight_decay=self.hparams.hpc_weight_decay,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)        
        # Architecture params
        parser.add_argument("--hpc_num_layers", default=6, type=int)
        parser.add_argument("--hpc_num_heads", default=4, type=int)
        parser.add_argument("--hpc_ffn_dim", default=300, type=int)
        parser.add_argument("--hpc_depthwise_conv_kernel_size",
                            default=11, type=int)
        parser.add_argument("--hpc_use_group_norm", default=1, type=int)
        parser.add_argument("--hpc_convolution_first", default=1, type=int)
        parser.add_argument("--hpc_dropout", default=0.2, type=float)

        # Multimodal args
        parser.add_argument("--hpc_get_emb", default=0, type=int)
        parser.add_argument("--hpc_emb_dim", default=256, type=int)

        # OPTIMIZER ARGS
        parser.add_argument("--hpc_learning_rate", default=0.000281, type=float)
        parser.add_argument("--hpc_weight_decay", default=0.008, type=float)

        # training specific (for this model)
        parser.add_argument("--hpc_data-type", type=str, default='HPC',
                            help='Possible values are HPC, PFC, all')

        return parser

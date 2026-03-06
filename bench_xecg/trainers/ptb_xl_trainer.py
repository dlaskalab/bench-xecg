from torch import nn
import torchmetrics
import torch

from .common_trainer import CommonTrainerDownstream
from ..utils.loss_utils import focal_loss


class TrainingPTB_XL(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.classification_task = config.classification_task
        self.num_classes = config.num_classes

        top_k = 1 if self.task == 'multiclass' else None

        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1, task=config.task, top_k=top_k)
        self.valid_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1,task=config.task, top_k=top_k)
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1,task=config.task, top_k=top_k)
        self.train_f1 = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task, top_k=top_k)
        self.valid_f1 = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1,task=config.task, top_k=top_k)
        self.test_f1_macro = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1,task=config.task, top_k=top_k)
        
        self.train_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)
        self.valid_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)
        self.test_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)

        self.train_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.valid_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.test_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)

    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds, targets)

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds, targets)

        self.log('train_loss', loss.detach().item(), prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=False)
        self.log('train_f1', self.train_f1, prog_bar=True)

        # auroc
        # self.train_auroc = self.train_auroc.cpu()
        self.train_auroc = self.train_auroc.to(logits.device)
        self.train_auroc(logits, targets.long())
        self.log("train_auroc", self.train_auroc)

        # auprc
        self.train_auprc = self.train_auprc.to(logits.device)
        self.train_auprc(logits, targets.long())
        self.log("train_auprc", self.train_auprc, prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.valid_acc = self.valid_acc.to(preds.device)
        self.valid_acc(preds, targets)

        self.valid_f1 = self.valid_f1.to(preds.device)
        self.valid_f1(preds, targets)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        self.log('val_acc', self.valid_acc, prog_bar=False)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets.long())
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        self.valid_auprc = self.valid_auprc.to(logits.device)
        self.valid_auprc(logits, targets.long())
        self.log('val_auprc', self.valid_auprc, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, targets)

        self.test_f1_macro = self.test_f1_macro.to(preds.device)
        self.test_f1_macro(preds, targets)

        self.log("test_loss", loss.detach().item())

        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1_macro)

        self.test_auroc = self.test_auroc.to(logits.device)
        self.test_auroc(logits, targets.long())
        self.log("test_auroc", self.test_auroc)

        self.test_auprc = self.test_auprc.to(logits.device)
        self.test_auprc(logits, targets.long())
        self.log("test_auprc", self.test_auprc)

        return loss
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['class_labels']

        genders = batch["genders"]
        ages = batch["ages"]

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x, age=ages, gender=genders)

        if self.task == 'multiclass':
            targets = torch.argmax(targets, dim=1)
            if self.use_focal_loss:
                loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.cross_entropy(logits, targets, weight=self.weights)
            preds = torch.argmax(logits, dim=1)

        elif self.task == 'multilabel':
            if self.use_focal_loss:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights)

            preds = (torch.sigmoid(logits) > 0.5).float()


        return loss, logits, preds, targets

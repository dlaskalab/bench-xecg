from typing import Any
import os
import pandas as pd
import numpy as np

from torch import nn
import torchmetrics
import torch
from torchmetrics import Metric
from torch import Tensor

from .common_trainer import CommonTrainerDownstream


class RegressionTrainer(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, target_key='age', weights=None, map_idx_dataloader=None, save_results_path=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.target_key = target_key
        self.map_idx_dataloader = map_idx_dataloader
        self.use_log = config.use_log
        self.save_results_path = save_results_path  # e.g. 'lightning_logs/train-age/<run_id>/<model_name>/results.csv'

        # --- MAE ---
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae  = torchmetrics.MeanAbsoluteError()

        # --- RSMAPE ---
        self.train_rsmape_0 = RobustMeanAbsoluteError(epsilon=0)
        self.valid_rsmape_0 = RobustMeanAbsoluteError(epsilon=0)
        self.test_rsmape_0  = RobustMeanAbsoluteError(epsilon=0)

        # --- MSE ---
        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.test_mse  = torchmetrics.MeanSquaredError()

        # --- R² ---
        self.train_r2 = torchmetrics.R2Score()
        self.valid_r2 = torchmetrics.R2Score()
        self.test_r2  = torchmetrics.R2Score()

        # --- Pearson ---
        self.train_pearson = torchmetrics.PearsonCorrCoef()
        self.valid_pearson = torchmetrics.PearsonCorrCoef()
        self.test_pearson  = torchmetrics.PearsonCorrCoef()

        # Buffers for CSV export (only populated when save_results_path is set)
        self._test_preds:   list[Tensor] = []
        self._test_targets: list[Tensor] = []
        self._test_dl_ids:  list[int]    = []   # now plain ints, no tensor needed
        self._dl_names: list[str]  = []

    def training_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.train_mae(preds, targets)
        self.log('train_mae', self.train_mae, prog_bar=True)
        self.train_mse(preds, targets)
        self.log('train_mse', self.train_mse, prog_bar=True)

        self.train_rsmape_0(preds, targets)
        self.log('train_rsmape_0', self.train_rsmape_0, prog_bar=False)

        self.train_r2(preds, targets)
        self.log('train_r2', self.train_r2, prog_bar=False)
        self.train_pearson(preds, targets)
        self.log('train_pearson', self.train_pearson, prog_bar=False)

        self.log('train_loss', loss.detach().item(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.valid_mae(preds, targets)
        self.log('valid_mae', self.valid_mae, prog_bar=True)
        self.valid_mse(preds, targets)
        self.log('valid_mse', self.valid_mse, prog_bar=True)

        self.valid_rsmape_0(preds, targets)
        self.log('valid_rsmape_0', self.valid_rsmape_0, prog_bar=False)

        self.valid_r2(preds, targets)
        self.log('valid_r2', self.valid_r2, prog_bar=False)
        self.valid_pearson(preds, targets)
        self.log('valid_pearson', self.valid_pearson,prog_bar=False)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        return loss
            
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        # Detect start of a new dataset
        if batch_idx == 0:
            print(f"Resetting metrics for dataloader {dataloader_idx}")
            self.test_mae.reset()
            self.test_mse.reset()
            self.test_rsmape_0.reset()
            self.test_r2.reset()
            self.test_pearson.reset()

        loss, preds, targets = self.predict_batch(batch)
        
        self.test_mae(preds, targets)
        self.test_mse(preds, targets)
        self.test_rsmape_0(preds, targets)
        self.test_r2(preds, targets)
        self.test_pearson(preds, targets)

        if self.save_results_path is not None:
            # records = batch['record']
            # print(f"[test_step] type={type(records)}, len={len(records)}, sample={records[0]}")  # debug
            self._test_preds.append(preds.detach().cpu())
            self._test_targets.append(targets.detach().cpu())
            # self._test_records.extend(records)                  # flat list of strings
            self._test_dl_ids.extend(
                np.arange(batch_idx * len(preds), (batch_idx + 1) * len(preds))
            )
            self._dl_names.extend(np.zeros(len(preds)) + dataloader_idx)

        # if the number of dataloader is bigger than 1
        if len(self.trainer.test_dataloaders) > 1:
            dataloader_idx = self.map_idx_dataloader[dataloader_idx] if self.map_idx_dataloader else dataloader_idx

            self.log(f'test_mae_{dataloader_idx}',      self.test_mae,        prog_bar=True)
            self.log(f'test_mse_{dataloader_idx}',      self.test_mse,        prog_bar=True)
            self.log(f'test_rsmape_0_{dataloader_idx}', self.test_rsmape_0,   prog_bar=False)

            self.log(f'test_r2_{dataloader_idx}',       self.test_r2,         prog_bar=False)
            self.log(f'test_pearson_{dataloader_idx}',  self.test_pearson,    prog_bar=False)

            self.log(f'test_loss_{dataloader_idx}',     loss.detach().item(), prog_bar=True)
            
        # if the number of dataloader is 1
        else:
            self.log('test_mae',        self.valid_mae,       prog_bar=True)
            self.log('test_mse',        self.valid_mse,       prog_bar=True)
            self.log('test_rsmape_0',   self.test_rsmape_0,   prog_bar=False)

            self.log('test_r2',         self.test_r2,         prog_bar=False)
            self.log('test_pearson',    self.test_pearson,    prog_bar=False)

            self.log('test_loss',       loss.detach().item(), prog_bar=True)
            
        return loss  
   
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch[self.target_key]
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        results = self.model(x).squeeze()
        if self.use_log:
            loss = nn.functional.mse_loss(results, torch.log(targets))
            results = torch.exp(results)
        else:
            loss = nn.functional.mse_loss(results, targets)

        return loss, results, targets
    
    def on_test_end(self):
        """Save predictions + targets to CSV if a path was provided."""
        if self.save_results_path is None or not self._test_preds:
            return

        all_preds   = torch.cat(self._test_preds).numpy()
        all_targets = torch.cat(self._test_targets).numpy()

        # Map numeric dataloader index → human-readable name
        dl_names = [
            (self.map_idx_dataloader or {}).get(int(i), str(int(i)))
            for i in self._dl_names
        ]

        df = pd.DataFrame({
            'id':   self._test_dl_ids,   # string identifier
            'dataset':    dl_names,
            'prediction': all_preds,
            'target':     all_targets,
        })

        os.makedirs(os.path.dirname(self.save_results_path), exist_ok=True)
        df.to_csv(self.save_results_path, index=False)
        print(f"Results saved to {self.save_results_path}")

        # Clear buffers
        self._test_preds.clear()
        self._test_targets.clear()
        self._test_dl_ids.clear()


class RobustMeanAbsoluteError(Metric):
    r"""`Computes e Robust Symmetric Mean Absolute Percentage Error`_ (RSMAPE):

    .. math:: \text{RSMAPE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} | / (|y_i| + |\hat{y_i}| + \epsilon)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        abs_err = torch.abs(preds - target)
        abs_pred = torch.abs(preds)
        abs_target = torch.abs(target)

        err = abs_err / (abs_pred + abs_target + self.epsilon)

        self.sum_abs_error += err.sum()
        self.total += abs_pred.numel()

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return self.sum_abs_error / self.total  

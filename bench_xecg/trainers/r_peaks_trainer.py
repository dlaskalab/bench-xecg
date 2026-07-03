import os
from typing import Optional

import torch
import numpy as np
from torch import nn
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.precision_recall
from torchmetrics import Metric
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from .common_trainer import CommonTrainerDownstream

def _make_binary_metrics() -> nn.ModuleDict:
    """Return a ModuleDict with the four per-split binary classification metrics."""
    return nn.ModuleDict({
        "rec":   torchmetrics.classification.BinaryRecall(),
        "f1":    torchmetrics.classification.BinaryF1Score(),
        "acc":   torchmetrics.classification.BinaryAccuracy(),
        "auprc": torchmetrics.classification.BinaryAveragePrecision(),
    })
 
 
def _make_distance_metrics(orig_freq: float, pred_freq: float) -> nn.ModuleDict:
    """Return RPeakDistanceMetrics for the four standard threshold windows."""
    windows = [150, 20, 10, 5]
    return nn.ModuleDict({
        str(w): RPeakDistanceMetric(orig_freq=orig_freq, pred_freq=pred_freq, threshold_window=w)
        for w in windows
    })


class TrainingRPeak(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None, save_results_path=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.sampling_freq = config.sampling_freq
        self.original_freq = config.original_freq
        self.plot_predictions = config.plot_predictions
        
        # One ModuleDict per split; Lightning registers them automatically.
        self.train_metrics = _make_binary_metrics()
        self.valid_metrics = _make_binary_metrics()
        self.test_metrics  = _make_binary_metrics()
 
        dist_kwargs = dict(orig_freq=self.original_freq, pred_freq=self.sampling_freq)
        self.val_distance_metrics  = _make_distance_metrics(**dist_kwargs)
        self.test_distance_metrics = _make_distance_metrics(**dist_kwargs)


    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
 
    def predict_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Run a forward pass and return (loss, probabilities_binary, targets, orig_peaks)."""
        x          = batch["signals"]
        r_peaks    = batch["r_peak"]       # [B, T]
        r_peaks_orig = batch["r_peak_orig"]
 
        if self.linear_probing:
            self.model.set_eval_linear_probing()
 
        logits = self.model(x).view(x.shape[0], -1)
 
        # Align lengths in case the model output differs slightly from the target.
        min_len = min(logits.shape[1], r_peaks.shape[1])
        logits  = logits[:, :min_len]
        r_peaks = r_peaks[:, :min_len]
 
        pos_weight = (
            torch.tensor(self.patch_size, dtype=torch.float32, device=logits.device)
            if self.weights is not None else None
        )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, r_peaks, pos_weight=pos_weight)
 
        preds_binary = (torch.sigmoid(logits) > 0.5).float()
        return loss, preds_binary, r_peaks, r_peaks_orig
 
    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
 
    def training_step(self, batch, _):
        loss, preds, targets, _ = self.predict_batch(batch)
        self._update_and_log_metrics(self.train_metrics, preds, targets, prefix="train")
        self.log("train_loss", loss.detach(), prog_bar=True)
        return loss
 
    def validation_step(self, batch, _):
        loss, preds, targets, r_peaks_orig = self.predict_batch(batch)
        self._update_and_log_metrics(self.valid_metrics, preds, targets, prefix="val")
        self._update_distance_metrics(self.val_distance_metrics, preds, r_peaks_orig)
        self.log("val_loss", loss.detach(), prog_bar=True)
        return loss
 
    def test_step(self, batch, _):
        loss, preds, targets, r_peaks_orig = self.predict_batch(batch)
        self._update_and_log_metrics(self.test_metrics, preds, targets, prefix="test")
        self._update_distance_metrics(self.test_distance_metrics, preds, r_peaks_orig)
        self.log("test_loss", loss.detach())
        return loss
 
    # ------------------------------------------------------------------
    # Epoch-end hooks
    # ------------------------------------------------------------------
 
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_distance_metrics(self.val_distance_metrics, prefix="val")
        self.plot_samples_if_needed(self.trainer.val_dataloaders, step="val")
 
    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._log_distance_metrics(self.test_distance_metrics, prefix="test")
        self.plot_samples_if_needed(self.trainer.test_dataloaders, step="test")


    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
 
    def _update_and_log_metrics(
        self,
        metrics: nn.ModuleDict,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
    ) -> None:
        metrics["rec"](preds, targets)
        metrics["f1"](preds, targets)
        metrics["acc"](preds, targets)
        metrics["auprc"](preds, targets.long())
 
        prog_bar_keys = {"rec", "f1", "acc", "auprc"}
        for name, metric in metrics.items():
            self.log(f"{prefix}_{name}", metric, prog_bar=(name in prog_bar_keys))
 
    @staticmethod
    def _update_distance_metrics(
        distance_metrics: nn.ModuleDict,
        preds: torch.Tensor,
        r_peaks_orig: list,
    ) -> None:
        for metric in distance_metrics.values():
            metric.update(preds, r_peaks_orig)
 
    def _log_distance_metrics(self, distance_metrics: nn.ModuleDict, prefix: str) -> None:
        primary_window = "150"
        for window_str, metric in distance_metrics.items():
            result = metric.compute()
            is_primary = window_str == primary_window
            if is_primary:
                self.log(f"{prefix}_avg_distance",       result["avg_distance"],       prog_bar=False)
                self.log(f"{prefix}_avg_distance_rp",    result["avg_distance_rp"],    prog_bar=False)
                self.log(f"{prefix}_avg_total_distance", result["avg_total_distance"], prog_bar=True)
            self.log(f"{prefix}_ppv_{window_str}", result["ppv"], prog_bar=False)
            self.log(f"{prefix}_tpr_{window_str}", result["tpr"], prog_bar=False)
            self.log(f"{prefix}_f1_{window_str}",  result["f1"],  prog_bar=is_primary)
            metric.reset()
 
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
 
    def plot_samples_if_needed(self, dataloader, step: str = "train") -> None:
        if not self.plot_predictions:
            return
        try:
            log_dir = (
                self.logger.log_dir
                if self.logger is not None and self.logger.log_dir is not None
                else "figs/"
            )
            for idx in range(2):
                sample = dataloader.dataset[idx]
                path = plot_r_peaks(
                    sample, self.model, self.sampling_freq,
                    self.device, log_dir, self.current_epoch,
                    f"r_peaks_{idx + 1}", step=step,
                )
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.log_image(key=f"reconstructions_{step}", images=[path])
        except Exception:
            import traceback
            traceback.print_exc()


# given all the heartbeats in the batch, find the closest predicted r-peak to each heartbeat and calculate a time distance

# ---------------------------------------------------------------------------
# Custom TorchMetric
# ---------------------------------------------------------------------------
 
class RPeakDistanceMetric(Metric):
    """
    Computes temporal distance and detection statistics between predicted
    R-peaks and ground-truth R-peaks.
 
    Distances are computed in seconds after converting both sets of indices
    from their respective sampling frequencies.  A prediction is counted as a
    true positive when it falls within ``threshold_window`` milliseconds of a
    ground-truth peak (symmetric window, so ± threshold_window / 2).
    """
 
    def __init__(
        self,
        orig_freq: float,
        pred_freq: float,
        threshold_window: float,          # milliseconds; required
        dist_sync_on_step: bool = False,
        process_group: Optional[object] = None,
        dist_sync_fn=None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
 
        self.orig_freq = orig_freq
        self.pred_freq = pred_freq
        # Half-window in seconds used for matching predictions to ground truth.
        self.half_window_s: float = (threshold_window / 1000.0) / 2.0
 
        self.add_state("total_distance",    default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_distance_rp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_predictions",   default=torch.tensor(0),   dist_reduce_fx="sum")
        self.add_state("true_positives",    default=torch.tensor(0),   dist_reduce_fx="sum")
        self.add_state("false_positives",   default=torch.tensor(0),   dist_reduce_fx="sum")
        self.add_state("positives",         default=torch.tensor(0),   dist_reduce_fx="sum")
 
    # ------------------------------------------------------------------
 
    def update(self, preds: torch.Tensor, r_peaks_orig: list) -> None:
        """
        Args:
            preds:        Binary predictions [B, T] with 1 at predicted R-peak positions.
            r_peaks_orig: Per-sample ground-truth peak indices in *original* frequency.
                          Padding values should be NaN.
        """
        for pred_row, orig_peaks in zip(preds, r_peaks_orig):
            gt_times   = self._gt_times(orig_peaks, pred_row.device)
            pred_times = self._pred_times(pred_row)
 
            n_gt   = gt_times.numel()
            n_pred = pred_times.numel()
 
            self.positives       += n_gt
            self.num_predictions += n_pred
 
            if n_pred == 0 or n_gt == 0:
                # All ground-truth peaks are unmatched; distances from predictions
                # to ground truth are undefined (or infinite) — skip accumulation.
                continue
 
            dist_matrix = torch.abs(pred_times.unsqueeze(1) - gt_times.unsqueeze(0))  # [P, G]
 
            # Distance from each *prediction* to its nearest ground-truth peak.
            min_dist_pred, nearest_gt_idx = dist_matrix.min(dim=1)   # [P]
            # Distance from each *ground-truth* peak to its nearest prediction.
            min_dist_gt, _                = dist_matrix.min(dim=0)    # [G]
 
            self.total_distance    += min_dist_pred.sum()
            self.total_distance_rp += min_dist_gt.sum()
 
            # True positives: unique ground-truth peaks matched within the window.
            matched_pred_mask = min_dist_pred <= self.half_window_s   # [P]
            matched_gt_ids    = nearest_gt_idx[matched_pred_mask]     # indices into gt_times
            self.true_positives  += matched_gt_ids.unique().numel()
            self.false_positives += (~matched_pred_mask).sum()
 
    # ------------------------------------------------------------------
 
    def compute(self) -> dict:
        avg_distance    = (self.total_distance    / self.num_predictions) * 1000  # → ms
        avg_distance_rp = (self.total_distance_rp / self.positives)       * 1000  # → ms
        avg_total       = (avg_distance + avg_distance_rp) / 2
 
        tp  = self.true_positives.float()
        fp  = self.false_positives.float()
        pos = self.positives.float()
 
        tpr = tp / pos             if pos > 0          else torch.tensor(0.0)
        ppv = tp / (tp + fp)       if (tp + fp) > 0    else torch.tensor(0.0)
        f1  = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) > 0 else torch.tensor(0.0)
 
        return {
            "avg_distance":       avg_distance,
            "avg_distance_rp":    avg_distance_rp,
            "avg_total_distance": avg_total,
            "total_predictions":  self.num_predictions,
            "total_matches":      self.true_positives,
            "ppv":  ppv,
            "tpr":  tpr,
            "f1":   f1,
        }
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _gt_times(self, orig_peaks, device: torch.device) -> torch.Tensor:
        """Convert raw ground-truth peak indices to seconds, dropping NaN padding."""
        return torch.tensor(
            [int(r) / self.orig_freq for r in orig_peaks if not torch.isnan(r)],
            dtype=torch.float32,
            device=device,
        )
 
    def _pred_times(self, pred_row: torch.Tensor) -> torch.Tensor:
        """Return predicted R-peak positions in seconds."""
        indices = torch.nonzero(pred_row, as_tuple=False).squeeze(-1).float()
        return indices / self.pred_freq


def plot_r_peaks(
    sample,
    model,
    sampling_freq: float,
    device: torch.device,
    logdir: str,
    epoch: int,
    name: str,
    step: str = "train",
) -> str:
    """Plot one ECG sample with predicted and ground-truth R-peaks overlaid."""
    max_samples = 20 * int(sampling_freq)
 
    with torch.no_grad():
        print(sample.keys())
        signal  = torch.tensor(sample["signals"]).unsqueeze(0).to(device)
        r_peaks = torch.tensor(sample["r_peak"]).unsqueeze(0).to(device)
 
        logits   = model(signal).view(signal.shape[0], -1)
        pred_bin = torch.sigmoid(logits) > 0.5
 
        # Limit to 10 seconds for readability.
        signal   = signal[:, :max_samples, :]
        pred_bin = pred_bin[:, :max_samples]
        r_peaks  = r_peaks[:, :max_samples]
 
    # Select a single lead for display (lead index 1, or the only lead).
    ecg = (
        signal[:, :, 1].cpu().squeeze().numpy()
        if signal.ndim > 2 and signal.shape[-1] > 1
        else signal.cpu().squeeze().numpy()
    )
 
    pred_indices = np.where(pred_bin.cpu().squeeze().numpy())[0]
    gt_indices   = np.where(r_peaks.cpu().squeeze().numpy())[0]
 
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(ecg)
 
    for i, peak in enumerate(pred_indices):
        ax.axvline(peak, color="darkorange", linestyle="solid",  linewidth=1.5,
                   label="Predicted R-peak" if i == 0 else "", alpha=0.5)
    for i, peak in enumerate(gt_indices):
        ax.axvline(peak, color="darkgreen",  linestyle="dashed", linewidth=1.5,
                   label="Ground Truth R-peak" if i == 0 else "", alpha=0.8)
 
    ax.set_xlabel("Timepoints", fontdict={"size": 24})
    ax.set_ylabel("Amplitude",  fontdict={"size": 24})
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.legend(fontsize=24)
    plt.tight_layout()
 
    out_dir = os.path.join(logdir, f"epoch_{epoch}", step)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path
import sys

import random

import torch
import torch.fft as fft

import numpy as np
from numbers import Real
from sklearn.utils import check_random_state

from typing import Any
import torch.nn as nn
from scipy.signal import butter, resample, sosfiltfilt, square
from typing import Any, Dict, List, Optional, Tuple, Union


class ECGPaperLayoutMask(nn.Module):
    """
    Randomly masks the ECG signal to simulate standard clinical print layouts 
    (3x4 and 2x6 formats).
    
    In these formats, not all leads are recorded simultaneously for the full duration.
    Instead, groups of leads are recorded for short segments of time.
    
    Attributes:
        p (float): Probability of applying the augmentation.
        format_ratio (float): Probability of choosing 2x6 format when mode is 'random'.
        mode (str): 'random' (chooses between 3x4 and 2x6), '3x4', or '2x6'.
        keep_rhythm_strip (bool): If True, one lead (defined by rhythm_lead_idx) 
                                  is kept for the full duration (simulating the bottom strip on a PDF).
        rhythm_lead_idx (int): Index of the lead to keep full length (usually 1 for Lead II).
    """
    def __init__(self, p=0.5, format_ratio=0.5, mode='random', keep_rhythm_strip=True, rhythm_lead_idx=1):
        super().__init__()
        self.p = p
        self.format_ratio = format_ratio
        self.mode = mode
        self.keep_rhythm_strip = keep_rhythm_strip
        self.rhythm_lead_idx = rhythm_lead_idx

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): ECG signal of shape (Batch, Leads, Time) or (Leads, Time).
                              Assumes 12 leads.
        """
        if np.random.rand(1) > self.p:
            return x

        # Create a mask of ones (keep everything by default)
        mask = np.zeros_like(x)
        
        # Iterate over batch to apply random choices per sample

        # Determine layout for this sample
        layout = self.mode

        if layout == 'random':
            layout = '2x6' if np.random.rand(1) < self.format_ratio else '3x4'
        
        # Initialize this sample's mask to zeros (we will fill in the 'visible' parts)
        
        if layout == '3x4':
            # 3x4 Format: 4 columns of time, 3 leads per column
            # Standard sequence: 
            # Col 1 (0-2.5s): Leads I, II, III (0,1,2)
            # Col 2 (2.5-5s): Leads aVR, aVL, aVF (3,4,5)
            # Col 3 (5-7.5s): Leads V1, V2, V3 (6,7,8)
            # Col 4 (7.5-10s): Leads V4, V5, V6 (9,10,11)
            
            segment_len = x.shape[0] // 4
            
            # Group 1 (0 - 0.25)
            mask[0:segment_len, 0:3] = 1
            # Group 2 (0.25 - 0.50)
            mask[segment_len:2*segment_len, 3:6] = 1
            # Group 3 (0.50 - 0.75)
            mask[2*segment_len:3*segment_len, 6:9] = 1
            # Group 4 (0.75 - 1.0)
            mask[3*segment_len:, 9:12] = 1
        elif layout == '2x6':
            # 2x6 Format: 2 columns of time, 6 leads per column
            # Col 1 (0-5s): Leads I-aVF (0-5)
            # Col 2 (5-10s): Leads V1-V6 (6-11)
            
            segment_len = x.shape[0] // 2
            
            # Group 1
            mask[0:segment_len, 0:6] = 1
            # Group 2
            mask[segment_len:, 6:12] = 1
        
        # Apply Rhythm Strip (keep Lead II visible for whole duration)
        if self.keep_rhythm_strip:
            mask[self.rhythm_lead_idx, :] = 1
        
        return x * mask

class Standardize:
    """Standardize the input sequence using NumPy."""
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Calculate mean and standard deviation using NumPy
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        
        # Initialize output array with zeros
        res = np.zeros_like(x)
        
        # Only perform division where scale is non-zero to avoid NaNs/Infs
        mask = scale != 0
        res = np.divide(x - loc, scale, out=res, where=mask)
        
        return res
    
class SOSFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x.T).T

class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')



def get_padding_mask(signal):
    """
        Get the padding mask for the signal.
        The mask is True for the padded values and False for the actual values.
    """
    return (signal != 0.).flip(1).cumsum(dim=1).flip(1) == 0

class RandomChangeAmplitude(nn.Module):
    """
        Randomly change the amplitude of the signal.
    """
    def __init__(self, amplitude_range=0.2, prob=1.0):
        super(RandomChangeAmplitude, self).__init__()
        self.amplitude_range = amplitude_range
        self.prob = prob

    def forward(self, signal):
        if self.prob == 0.: return signal
        
        scale = (np.random.rand() - 0.5) * self.amplitude_range + 1
        return signal * scale
        

class RandomShiftBaselineWander(nn.Module):
    """
        Randomly shift the baseline wander.
    """
    def __init__(self, signal_fs=500, cutoff_freq=0.5):
        super().__init__()
        self.signal_fs = signal_fs
        self.cutoff_freq = cutoff_freq

    def forward(self, signal):
        baseline_wander = extract_baseline_fft_torch(signal, self.cutoff_freq, self.signal_fs).squeeze()
        # get randint to shift the signal based on sig len
        shift = np.random.randint(0, signal.shape[0] - 1)
        baseline_shifted = torch.roll(baseline_wander, shifts=shift, dims=0)
        signal = signal - baseline_wander + baseline_shifted
        return signal

class RandomSwitchBaselineWanderBatched(nn.Module):
    """
        Randomly switch the baseline wander in a batch of signals.
    """
    def __init__(self, signal_fs=500, cutoff_freq=0.5):
        super().__init__()
        self.signal_fs = signal_fs
        self.cutoff_freq = cutoff_freq

    def forward(self, signals):
        if signals.shape[0] == 1:
            # if the batch size is 1, just return the signal
            return signals
        
        # get the baseline wander
        baseline_wander = extract_baseline_fft_torch(signals, self.cutoff_freq, self.signal_fs)
        shift = np.random.randint(0, signals.shape[1] - 1)
        baseline_shifted = torch.roll(baseline_wander, shifts=shift, dims=1)

        # get the batch size
        batch_size = signals.shape[0]
        # get the random index to switch the baseline wander
        rand_idxs = torch.randperm(batch_size, device=signals.device)
        # switch the baseline wander
        zeroed_mask = (signals == 0).all(dim=1, keepdim=True)  # Mask for any zeroed-out channels
        padding_mask = get_padding_mask(signals)
        mask = ~padding_mask & ~zeroed_mask

        signals_without_baseline = signals - baseline_wander + baseline_shifted[rand_idxs]
        signals = signals_without_baseline * mask
        return signals
         
def extract_baseline_fft_torch(ecg: torch.Tensor, cutoff_freq=0.5, signal_fs=500) -> torch.Tensor:
    """
    Efficient baseline wander extraction using FFT in pure PyTorch.

    Args:
        ecg: Tensor of shape [sig_len, num_leads]
        fs: Sampling frequency in Hz (default: 500)
        cutoff: Lowpass cutoff frequency in Hz (default: 0.5)

    Returns:
        baseline: Tensor of same shape as ecg, containing only the low-frequency components
    """
    if len(ecg.shape) == 2:
        # batched data
        ecg = ecg.unsqueeze(0)  # Add batch dimension

    _, sig_len, _ = ecg.shape
    device = ecg.device

    # Compute FFT
    ecg_fft = torch.fft.rfft(ecg, dim=1)

    # Compute frequency bins
    freqs = torch.fft.rfftfreq(sig_len, d=1/signal_fs).to(device)  # Shape: [rfft_len]

    # Create lowpass mask: freqs <= cutoff
    mask = (freqs <= cutoff_freq).unsqueeze(0).unsqueeze(2)  # Shape: [rfft_len, 1] to broadcast

    # Apply lowpass filter
    filtered_fft = ecg_fft * mask

    # Inverse FFT to get baseline
    baseline = torch.fft.irfft(filtered_fft, n=sig_len, dim=1)

    return baseline

class Normalize(nn.Module):
    """
        Normalize the signal.
    """
    def __init__(self):
        super(Normalize, self).__init__()


    def forward(self, signal):
        std = signal.std(axis=(0))
        std[std == 0] = 1 # avoid division by zero, samples with std = 0 are all zero
        return (signal - signal.mean(axis=(0))) / std
    
class NormalizePadded(nn.Module):
    """
        Normalize the signal.
    """
    def __init__(self):
        super(NormalizePadded, self).__init__()

    def forward(self, signal):
        # we should have one end and one start for each lead
        mask = (np.abs(signal) >= 1e-4).astype(np.float32)
        # set false to nan values
        mask_start = mask.cumsum(0)
        mask_start[mask_start == 0] = np.inf
        start = mask_start.argmin(0)

        mask_end = np.flip(np.flip(mask, axis=0).cumsum(0), axis=0)
        mask_end[mask_end == 0] = np.inf
        end = mask_end.argmin(0)

        # if start !+ all zero print
        # if (start != 0).any():
        #     print('start of real signal:', start)
        # if ((end != signal.shape[0] - 1) & (end != 0)).any():
        #    print('end of real signal:', end)

        # Create range tensor: (time, 1)
        time_range = np.arange(signal.shape[0])[:, np.newaxis]

        start = np.atleast_1d(start)
        start_broadcast = start[np.newaxis, :]
        end = np.atleast_1d(end)
        end_broadcast = (end + 1)[np.newaxis, :]

        valid_mask = (time_range >= start_broadcast) & (time_range < end_broadcast)
        valid_mask = valid_mask.astype(np.float32) # Ensure mask is float for multiplication

        # Create valid mask: (time, num_leads)
        # valid_mask = (time_range >= start.unsqueeze(0)) & (time_range < end.unsqueeze(0))

        # Count valid samples per lead
        count = valid_mask.sum(axis=0, keepdims=True)  # (1, num_leads)
        count = np.clip(count, a_min=1, a_max=None)  # avoid division by zero

        # Compute mean over valid regions
        masked_signal = signal * valid_mask
        mean = masked_signal.sum(axis=0, keepdims=True) / count  # (1, num_leads)

        # Compute std over valid regions
        diff = (signal - mean) * valid_mask
        var = (diff ** 2).sum(axis=0, keepdims=True) / count  # (1, num_leads)
        std = np.sqrt(var)
        std = np.where(std == 0, np.ones_like(std), std)  # replace 0 with 1
        # print('mean, std:', mean, std)

        # Normalize only valid regions
        normalized = (signal - mean) / std
        signal = np.where(valid_mask, normalized, signal)

        return signal

    
class RandomCrop(nn.Module):
    """
        Randomly crop the signal.
    """
    def __init__(self, crop_size=0.9, max_length=None):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.max_length = max_length

    def forward(self, signal):
        # Get the size of the signal
        end = np.flip(np.flip((signal != 0), axis=0).cumsum(0), axis=0).max(axis=-1).max(axis=-1)
        # start of signal: there may be padding at the beginning of the signal
        start = (signal != 0).cumsum(axis=0).max(axis=-1).max(axis=-1)

        if not (signal[0] == 0).all():
            start = 0 # if the signal do not starts with zeros, we can start from the beginning

        # Calculate the target length
        # consider a maximun length of the signal
        signal_length = min(end - start, self.max_length) # Ensure we don't exceed the actual length
        
        target_length = int(np.floor(signal_length * self.crop_size))
        # Randomly sample the starting point for the cropping (cut-off)

        if signal_length <= target_length:
            # not enough real (non-padding) signal to satisfy the crop ratio
            # (e.g. an all-zero/degenerate record) - skip cropping
            return signal

        start_idx = np.random.randint(low=start, high=signal_length - target_length + start)
        # Crop the signal

        # print('RandomCrop: crop_size =', self.crop_size, ', target_length =', target_length, ', start_idx =', start_idx, 'shape =', signal[start_idx:start_idx + target_length, ...].shape)
        return signal[start_idx:start_idx + target_length, ...]
    
class CropFixedLen(nn.Module):
    """
        Randomly crop the signal.
    """
    def __init__(self, length):
        super(CropFixedLen, self).__init__()
        self.length = length

    def forward(self, signal):
        if self.length is None:
            return signal
        # Get the size of the signal
        if signal.shape[0] > self.length:
            # ('Cropping signal from length', signal.shape[0], 'to', self.length, 'shape:', signal[:self.length, ...].shape)
            return  signal[:self.length, ...]
        
        return signal


class RandomDropLeads(nn.Module):
    """
        Randomly drop leads from the signal.
    """
    def __init__(self, probability=0.5, keep_lead_II=True):
        super(RandomDropLeads, self).__init__()
        self.probability = probability
        self.keep_lead_II = keep_lead_II  # Ensure lead II is never removed

    def forward(self, signal):
        if self.training and self.probability > 0: # Also check if probability is non-zero
            # Create a copy to avoid modifying the original tensor inplace
            
            # Determine leads to remove (ensure consistent device if signal is on GPU)
            leads_to_remove = np.random.random(signal.shape[-1]) < self.probability
            # leads_to_remove = torch.from_numpy(leads_to_remove_np).to(signal.device) # Convert to tensor and move to correct device

            # Ensure lead II (index 1 assuming standard 12-lead) is never removed
            if self.keep_lead_II: # Check if there's more than one lead
                leads_to_remove[1] = False
            else:
                # ensure at least one lead is kept
                if leads_to_remove.sum() == signal.shape[-1]:
                    # select one random lead to keep, this avoids removing all leads
                    random_lead = np.random.randint(0, signal.shape[-1])
                    leads_to_remove[random_lead] = False
                
            signal_out = signal.copy()
            signal_out[:, leads_to_remove] = 0

            # ('RandomDropLeads shape:', signal_out.shape, 'dropped leads:', torch.where(leads_to_remove)[0].cpu().numpy())
            return signal_out
        else:
            # If not training or probability is 0, return the original signal
            return signal
    
class Jitter(object):
    """
        Add gaussian noise to the sample.
    """
    def __init__(self, sigma=0.2, amplitude=0.6, prob=1.0) -> None:
        self.sigma = sigma
        self.amplitude = amplitude
        self.prob = prob

    def __call__(self, sample) -> Any:
        # 0. If the probability is 0, return the original sample.
        if self.prob == 0. or np.random.uniform() > self.prob: return sample

        # 3. Generate noise for the *entire* batch.
        noise = np.random.randn(*sample.shape) * self.sigma

        # 4. Calculate the amplitude scaling for the *entire* batch.
        amplitude_scaling = self.amplitude * sample

        # 5. Apply the jitter only where the mask is True. This is done using element-wise multiplication and addition.
        jittered_tensor = sample + amplitude_scaling * noise

        return jittered_tensor

class FTSurrogate(object):
     """
     FT surrogate augmentation of a single EEG channel, as proposed in [1]_.
     Code (modified) from https://github.com/braindecode/braindecode/blob/master/braindecode/augmentation/functional.py 
     
 
     Parameters
     ----------
     X : torch.Tensor
         EEG input example.
     phase_noise_magnitude: float
         Float between 0 and 1 setting the range over which the phase
         pertubation is uniformly sampled:
         [0, `phase_noise_magnitude` * 2 * `pi`].
     channel_indep : bool
         Whether to sample phase perturbations independently for each channel or
         not. It is advised to set it to False when spatial information is
         important for the task, like in BCI.
     random_state: int | numpy.random.Generator, optional
         Used to draw the phase perturbation. Defaults to None.
 
     Returns
     -------
     torch.Tensor
         Transformed inputs.
     torch.Tensor
         Transformed labels.
 
     References
     ----------
     .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
        Clifford, G. D. (2018). Addressing Class Imbalance in Classification
        Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
        preprint arXiv:1806.08675.
     """
     def __init__(self, phase_noise_magnitude, channel_indep=False, seed=None, prob=1.0) -> None:
         self.phase_noise_magnitude = phase_noise_magnitude
         self.channel_indep = channel_indep
         self.seed = seed
         self.prob = prob
         self._new_random_fft_phase = {
             0: self._new_random_fft_phase_even,
             1: self._new_random_fft_phase_odd
         }
 
     def _new_random_fft_phase_odd(self, c, n, device='cpu', seed=None):
         rng = check_random_state(seed)
         random_phase = torch.from_numpy(
             2j * np.pi * rng.random((c, (n - 1) // 2))
         ).to(device)
 
         return torch.cat([
             torch.zeros((c, 1), device=device),
             random_phase,
             -torch.flip(random_phase, [-1]).to(device=device)
         ], dim=-1)
     
     def _new_random_fft_phase_even(self, c, n, device='cpu', seed=None):
         rng = check_random_state(seed)
         random_phase = torch.from_numpy(
             2j * np.pi * rng.random((c, n // 2 - 1))
         ).to(device)
 
         return torch.cat([
             torch.zeros((c, 1), device=device),
             random_phase,
             torch.zeros((c, 1), device=device),
             -torch.flip(random_phase, [-1]).to(device=device)
         ], dim=-1)
 
     def __call__(self, sample) -> Any:
         if np.random.uniform() < self.prob:
             assert isinstance(
                 self.phase_noise_magnitude,
                 (Real, torch.FloatTensor, torch.cuda.FloatTensor)
             ) and 0 <= self.phase_noise_magnitude <= 1, (
                 f"eps must be a float beween 0 and 1. Got {self.phase_noise_magnitude}."
             )
 
             f = fft.fft(sample.double(), dim=-1)
 
             n = f.shape[-1]
             random_phase = self._new_random_fft_phase[n % 2](
                 f.shape[-2] if self.channel_indep else 1,
                 n,
                 device=sample.device,
                 seed=self.seed
             )
 
             if not self.channel_indep:
                 random_phase = torch.tile(random_phase, (f.shape[-2], 1))
 
             if isinstance(self.phase_noise_magnitude, torch.Tensor):
                 self.phase_noise_magnitude = self.phase_noise_magnitude.to(sample.device)
 
             f_shifted = f * torch.exp(self.phase_noise_magnitude * random_phase)
             shifted = fft.ifft(f_shifted, dim=-1)
             sample_transformed = shifted.real.float()

             return sample_transformed
 
         else:
             return sample

class RandomResample(torch.nn.Module):
    def __init__(self, current_freq=360, max_freq_delta_ratio=0.05):
        super().__init__()
        self.current_freq = current_freq
        self.max_freq_delta = int(current_freq * max_freq_delta_ratio)

    def forward(self, signal):
        freq_delta = torch.randint(-self.max_freq_delta, self.max_freq_delta + 1, (1,)).item()
        target_freq = self.current_freq + freq_delta
        return resample_signal(signal, self.current_freq, target_freq)
    

def resample_signal(signal: torch.Tensor, current_freq: float = 500, target_freq: float = 400):
    """
    Resample a tensor of shape (C, L) to the target frequency.

    Args:
        signal (torch.Tensor): A tensor of shape (C, L),
        where C is the number of channels and L is the length of the signal.
        current_freq (float): The current frequency of the signal in Hz.
        target_freq (float): The desired frequency in Hz.

    Returns:
        torch.Tensor: A tensor of shape (C, new_L) resampled to the target frequency.
    """
    signal_length, _  = signal.shape
    target_length = int(signal_length * target_freq / current_freq)
    return resample(signal, target_length)
    
class Rescaling(object):
    """
        Randomly rescale features of the sample.
    """
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma

    def __call__(self, sample) -> Any:
        sample = sample * torch.normal(mean=torch.Tensor([1]), std=torch.Tensor([self.sigma]))
        return sample


class Shift(object):
    """
        Randomly shift the signal in the time domain.
    """
    def __init__(self, fs=360, padding_len_sec=1) -> None:
        self.padding_len = fs * padding_len_sec # padding len in ticks 

    def __call__(self, sample) -> Any:
        # define padding size 
        left_pad = int(torch.rand(1) * self.padding_len)
        right_pad = self.padding_len - left_pad

        # zero-pad the sample
        # note: the signal length is now extended by self.padding_len
        padded_sample = torch.nn.functional.pad(sample, (left_pad, right_pad), value=0)

        # get back to the original signal length
        if torch.rand(1) < 0.5:
            return padded_sample[..., :sample.shape[-1]]
        else:
            return padded_sample[..., right_pad:sample.shape[-1]+right_pad]

class CropResizing(object):
    """
        Randomly crop the sample and resize to the original length.
    """
    def __init__(self, lower_bnd=0.8, upper_bnd=0.8, fixed_crop_len=None, start_idx=None, resize=False, fixed_resize_len=None) -> None:
        self.lower_bnd = lower_bnd
        self.upper_bnd = upper_bnd
        self.fixed_crop_len = fixed_crop_len
        self.start_idx = start_idx
        self.resize = resize
        self.fixed_resize_len = fixed_resize_len

    def __call__(self, sample) -> Any:
        sample_dims = sample.dim()
        
        # define crop size
        if self.fixed_crop_len is not None:
            crop_len = self.fixed_crop_len
        else:
            # randomly sample the target length from a uniform distribution
            crop_len = int(sample.shape[0]*np.random.uniform(low=self.lower_bnd, high=self.upper_bnd))
        
        # define cut-off point
        if self.start_idx is not None:
            start_idx = self.start_idx
        else:
            # randomly sample the starting point for the cropping (cut-off)
            try:
                start_idx = np.random.randint(low=0, high=sample.shape[0]-crop_len)
            except ValueError:
                # if sample.shape[-1]-crop_len == 0, np.random.randint() throws an error
                start_idx = 0

        # crop and resize the signal
        if self.resize == True:
            # define length after resize operation
            if self.fixed_resize_len is not None:
                resize_len = self.fixed_resize_len
            else:
                resize_len = sample.shape[-1]

            # crop and resize the signal
            cropped_sample = torch.zeros_like(sample[..., :resize_len])
            if sample_dims == 2:
                for ch in range(sample.shape[-2]):
                    resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[ch, start_idx:start_idx+crop_len])
                    cropped_sample[ch, :] = torch.from_numpy(resized_signal)
            elif sample_dims == 3:
                for f_bin in range(sample.shape[-3]):
                    for ch in range(sample.shape[-2]):
                        resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[f_bin, ch, start_idx:start_idx+crop_len])
                        cropped_sample[f_bin, ch, :] = torch.from_numpy(resized_signal)
            else:
                sys.exit('Error. Sample dimension does not match.')
        else:
            # only crop the signal
            cropped_sample = torch.zeros_like(sample)
            cropped_sample = sample[start_idx:start_idx+crop_len, ...]

        # print('CropResizing: crop_len =', crop_len, ', start_idx =', start_idx, 'shape =', cropped_sample.shape)

        return cropped_sample

class Interpolation(object):
    """
        Undersample the signal and interpolate to initial length.
    """
    def __init__(self, step=2, prob=1.0) -> None:
        self.step = step
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            sample_sub = sample[..., ::self.step]
            sample_interpolated = np.ones_like(sample)
            
            sample_dims = sample.dim()
            if sample_dims == 2:
                for ch in range(sample.shape[-2]):
                    sample_interpolated[ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[ch])
            elif sample_dims == 3:
                for f_bin in range(sample.shape[-3]):
                    for ch in range(sample.shape[-2]):
                        sample_interpolated[f_bin, ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[f_bin, ch])
            else:
                sys.exit('Error. Sample dimension does not match.')

            return torch.from_numpy(sample_interpolated)
        else:
            return sample

class Masking(object):
    """
        Randomly zero-mask the sample.
        Got this from https://stackoverflow.com/questions/70092136/how-do-i-create-a-random-mask-matrix-where-we-mask-a-contiguous-length
        Don't touch the code!
    """
    def __init__(self, factor:float=0.75, fs:int=360, patch_size_sec:float=1, masking_mode="random", prob=1.0) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = int(patch_size_sec * fs)   # patch_size[ticks] = patch_size[sec] * fs[Hz]
        self.masking_mode = masking_mode
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            # create the mask
            mask = torch.ones_like(sample)

            # determine the number of patches to be masked
            nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
            
            indices_weights = np.random.random((mask.shape[0], nb_patches + 1))

            number_of_ones = mask.shape[-1] - self.patch_size * nb_patches

            ones_sizes = np.round(indices_weights[:, :nb_patches].T
                                * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)
            ones_sizes[:, 1:] += self.patch_size

            zeros_start_indices = np.cumsum(ones_sizes, axis=-1)

            if self.masking_mode == "block":
                for zeros_idx in zeros_start_indices[0]:
                    mask[..., zeros_idx: zeros_idx + self.patch_size] = 0
            else:
                for sample_idx in range(len(mask)):
                    for zeros_idx in zeros_start_indices[sample_idx]:
                        mask[sample_idx, zeros_idx: zeros_idx + self.patch_size] = 0

            return sample * mask
        else:
            return sample
    
    
class FrequencyShift(object):
    """
    Adds a shift in the frequency domain to all channels.
    Note that here, the shift is the same for all channels of a single example.
    Code (modified) from https://github.com/braindecode/braindecode/blob/master/braindecode/augmentation/functional.py

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    delta_freq : float
        The amplitude of the frequency shift (in Hz).
    sfreq : float
        Sampling frequency of the signals to be transformed.
    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    def __init__(self, delta_freq=0, s_freq=360, prob=1.0) -> None:
        self.delta_freq = delta_freq
        self.s_freq = s_freq
        self.prob = prob

    def _analytic_transform(self, X):
        if torch.is_complex(X):
            raise ValueError("X must be real.")

        N = X.shape[-1]
        f = fft.fft(X, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2

        return fft.ifft(f * h, dim=-1)

    def _nextpow2(self, n):
        """Return the first integer N such that 2**N >= abs(n)."""

        return int(np.ceil(np.log2(np.abs(n))))

    def _frequency_shift(self, X, fs, f_shift):
        """
        Shift the specified signal by the specified frequency.
        See https://gist.github.com/lebedov/4428122
        """
        nb_channels, N_orig = X.shape[-2:]

        # Pad the signal with zeros to prevent the FFT invoked by the transform
        # from slowing down the computation:
        N_padded = 2 ** self._nextpow2(N_orig)
        t = torch.arange(N_padded, device=X.device) / fs
        padded = torch.nn.functional.pad(X, (0, N_padded - N_orig))

        analytical = self._analytic_transform(padded)
        if isinstance(f_shift, (float, int, np.ndarray, list)):
            f_shift = torch.as_tensor(f_shift).float()

        reshaped_f_shift = f_shift.repeat(N_padded, nb_channels).T
        shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)

        return shifted[..., :N_orig].real.float()

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            sample_transformed = self._frequency_shift(
                X=sample,
                fs=self.s_freq,
                f_shift=self.delta_freq,
            )

            return sample_transformed
        else:
            return sample
    
class TimeFlip(object):
    """
        Flip the signal vertically.
    """
    def __init__(self, prob=1.0) -> None:
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            return torch.flip(sample, dims=[-1])
        else:
            return sample
    
class SignFlip(object):
    """
        Flip the signal horizontally.
    """
    def __init__(self, prob=1.0) -> None:
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            return -1*sample
        else:
            return sample
        
class SpecAugment(object):
    """
        Randomly masking frequency or time bins of signal's short-time Fourier transform.
        See https://arxiv.org/pdf/2005.13249.pdf
    """
    def __init__(self, masking_ratio=0.2, n_fft=120) -> None:
        self.masking_ratio = masking_ratio
        self.n_fft = n_fft

    def __call__(self, sample) -> Any:
        sample_dim = sample.dim()

        if sample_dim < 3:
            masked_sample = self._mask_spectrogram(sample)
        elif sample_dim == 3:
            # perform masking separately for all entries in the first dimension 
            # and eventually concatenate the masked entries to retrieve the intial shape 
            masked_sample = torch.Tensor()
            for i in range(sample.shape[0]):
                masked_sub_sample = self._mask_spectrogram(sample[i])
                masked_sample = torch.cat((masked_sample, masked_sub_sample.unsqueeze(0)), dim=0)
        else: 
            print(f"Augmentation was not built for {sample_dim}-D input")

        return masked_sample

    def _mask_spectrogram(self, sample):
        sample_length = sample.shape[-1]

        # compute the Fourier transform
        spec = torch.stft(sample, n_fft=self.n_fft, return_complex=True)

        if random.random() < 0.5:
            # frequency domain
            masked_block_size = int(spec.shape[-2]*self.masking_ratio)
            start_idx = random.randint(0, spec.shape[-2] - masked_block_size)
            end_idx = start_idx + masked_block_size

            # mask the bins
            spec[..., start_idx:end_idx, :] = 0.+0.j
        else:
            # time domain
            masked_block_size = int(spec.shape[-1]*self.masking_ratio)
            start_idx = random.randint(0, spec.shape[-1] - masked_block_size)
            end_idx = start_idx + masked_block_size

            # mask the bins
            spec[..., start_idx:end_idx] = 0.+0.j

        # perform the inverse Fourier transform to get the new signal
        masked_sample = torch.istft(spec, n_fft=self.n_fft, length=sample_length)

        return masked_sample
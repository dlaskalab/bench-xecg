import os
import random

import torch
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from tqdm import tqdm

from .generic_utils import RandomSwitchBaselineWanderBatched


class ECGSleepApneaDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', augmentations=None, split_seed=None):
        self.patch_size = config.patch_size
        self.sampling_freq = config.sampling_freq
        self.data_folder = config.data_folder_sleep_apnea
        self.split = split
        self.split_seed = split_seed
        self.leads = config.leads
        self.window_size = config.window_size * 100 # seconds of data to classify
        self.context_size_half = config.context_size * 100 // 2 # seconds of context for each ecg
        self.skip_ignored_indexes = config.skip_ignored_indexes

        self.augmentations = augmentations
        self.segment_size = 6000  # 60 seconds in samples

        if self.sampling_freq % self.patch_size != 0:
            print(f"Warning: Sampling freq {self.sampling_freq} should be divisible by patch size {self.patch_size}")
        
        if self.window_size > self.segment_size:
            print(f"Warning: Window size {self.window_size} is larger than segment size {self.segment_size}, this may lead to unexpected behavior.")

        self.load_records()
        self.load_samples()

    def load_records(self):
        # list os files from the RECORDS file
        with open(os.path.join(self.data_folder, 'RECORDS'), 'r') as f:
            records = f.readlines()

        # keep only the first 3 characters of each line and sort deterministically before splitting
        records = sorted(set(record.strip()[:3] for record in records if record.strip()))

        train_val_records = [r for r in records if not r.startswith('x')]
        test_records = [r for r in records if r.startswith('x')]

        if self.split_seed is not None:
            random.Random(self.split_seed).shuffle(train_val_records)

        split_idx = int(len(train_val_records) * 0.8)
        if self.split == 'train':
            selected = train_val_records[:split_idx]
            print(f"Training records (seed={self.split_seed}): {selected}")
        elif self.split == 'val':
            selected = train_val_records[split_idx:]
            print(f"Validation records (seed={self.split_seed}): {selected}")
        elif self.split == 'test':
            selected = test_records
            print(f"Test records: {selected}")
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.records = [os.path.join(self.data_folder, record) for record in selected]    

    def load_samples(self):
        # for each record, divide it into 5 min segments
        # and save the segments in a list
        self.samples = []
        self.annotations = []
        self.segment_id = []

        # for each ecg record, read the signal and annotations
        for record in tqdm(self.records, desc='Loading samples'):
            # read the record
            signal, _ = wfdb.rdsamp(record)
            ann = wfdb.rdann(record, 'apn')

            # get the length of the signal (considering it only as a multiple of segment_size)
            length = len(signal) - (len(signal) % self.window_size)

            # divide the signal into window_size segments
            count = 0
            if self.window_size < self.segment_size: count_2 = 0

            for i in range(0, length, self.window_size):
                if signal.shape[0] - i < self.window_size:
                    break # do not use the last segment if it is smaller than window_size
                
                # append the segment  of window_size to the samples list
                if self.context_size_half > 0:
                    # add context before and after the segment
                    start = max(0, i - self.context_size_half)
                    end = min(length, i + self.window_size + self.context_size_half)
                    sample = signal[start:end]
                    # print(f"Original sample shape: {signal.shape}, start: {start}, end: {end}, segment shape: {sample.shape}")

                    # if the context size is larger than the segment size, pad the sample using numpy
                    if start == 0:
                        pad_len = self.context_size_half - i
                        if pad_len > 0:
                            pad_array = np.zeros((pad_len, signal.shape[1]))
                            sample = np.concatenate((pad_array, sample), axis=0)
                    if end == length:
                        pad_len = self.context_size_half - (length - (i + self.window_size))
                        if pad_len > 0:
                            pad_array = np.zeros((pad_len, signal.shape[1]))
                            sample = np.concatenate((sample, pad_array), axis=0)
                else:
                    sample = signal[i:min(length, i + self.window_size)]


                # get the annotations for the segment
                annotations = []
                while count < len(ann.sample) and ann.sample[count] < i + self.window_size:
                    annotations.append(1. if ann.symbol[count] == 'A' else 0.)
                    self.segment_id.append((record, count))
                    if self.window_size < self.segment_size: 
                        count_2 += 1
                        # case of transformer where segment size is smaller than window size (10 seconds)
                        if count_2  % (self.segment_size // self.window_size) == 0:
                            count += 1
                        break
                    else:
                        count += 1

                        
                if len(annotations) > 0:
                    self.samples.append(sample)
                    self.annotations.append(torch.tensor(annotations))

            # print(f"Loaded {len(self.samples)} segments from record {record}")
        print(f"Total samples: {len(self.samples)}, total annotations: {len(self.annotations)}, total segment ids: {len(self.segment_id)}")
                # else:
                    # print(f"Warning: No annotations found for segment {i} in record {record}, skipping... (This should not happen)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = self.annotations[idx]

        sample = self.resample_if_needed(sample, {'fs': 100})  # original sampling freq is 100 Hz

        # map to the correct lead
        tensor = torch.zeros(sample.shape[0], len(self.leads), dtype=torch.float32)
        if len(self.leads) == 1:
            tensor[:, 0] = sample[:, 0]  # ECG
        else:
            tensor[:, 1] = sample[:, 0]  # ECG

        if self.augmentations is not None:
            tensor = self.augmentations(tensor)

        return {
            'signal': tensor,
            'annotation': ann,
            'segment_id': self.segment_id[idx],
        }
    
    def resample_if_needed(self, signal, info):
        if self.sampling_freq != info['fs']:
            signal = nk.signal_resample(signal, sampling_rate=info['fs'], desired_sampling_rate=self.sampling_freq, method='FFT')   
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal
    

def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        labels = [item['annotation'] for item in batch]
        segment_ids = [item['segment_id'] for item in batch]

        # pad to same length and pad to match the patch size module
        if config.shuffle_baseline_wander_in_batch and split == 'train':
            signals = baseline_shuffler(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True))
        else:
            signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

        labels = torch.tensor(np.array([label.numpy() for label in labels]), dtype=torch.float32)

        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
            
        return {
            'signals': signals,
            'labels': labels,
            'segment_ids': segment_ids,
        }

    return collate_fn
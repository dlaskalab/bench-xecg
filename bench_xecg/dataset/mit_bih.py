import os
import random
from typing_extensions import override

import torch
import wfdb
import neurokit2 as nk
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .generic_utils import RandomSwitchBaselineWanderBatched

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
conversion = {
    'MLII' : 'II',
}

train = [101, 106, 108, 109, 112, 115, 116, 118, 119, 122, 124, 201, 205, 207, 208, 209, 215, 220, 223, 230]
test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
val = [203, 114]

def convert_label(symbol):
    """
    Convert the symbols to the main class label
    """
    if symbol in ['N', 'L', 'R', 'e', 'j']:
        return 'N'
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'  # Supraventricular ectopic
    elif symbol in ['V', 'E']:
        return 'V'  # Ventricular ectopic
    elif symbol in ['F']:
        return 'F'  # Fusion
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'  # Unknown
    else:
        print(symbol)
        raise (f'Unknown symbol {symbol}')
    
valid_annotations = set(['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'])

class ECGMITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        
        self.data_folder = config.data_folder_mit
        self.split = split
        self.samples = []
        self.patch_size = config.patch_size
        self.num_classes = config.num_classes 

        # defined in number of timepoints of model's frequency
        self.win_len = config.win_len 
        self.skip_majority_class_samples = config.skip_majority_class_samples
        self.split_val_by_patient = config.split_val_by_patient
        self.augmentations = augmentations

        # model's sampling freq
        self.sampling_freq = config.sampling_freq
        self.leads_to_use = config.leads
        self.context_len = config.context_len
        self.original_freq = 360
        self.freq_factor = self.sampling_freq / self.original_freq
        self.r_peaks_detection = config.r_peaks_detection
        self.random_shift = config.random_shift
        self.extend_labels = config.extend_labels
        
        print(f"freq_factor: {self.freq_factor}, sampling_freq: {self.sampling_freq}, original_freq: {self.original_freq}")

        self.load_patient_data(split)
        self.load_samples(split)


    def load_patient_data(self, subset):
        if self.split_val_by_patient:
            self.patients = train if subset == 'train' else val if subset == 'val' else test
        else:   
            self.patients = train + val if subset == 'train' else test if subset == 'test' else val

        self.headers = {}
        self.signals = {}
        self.r_peaks = {}
        self.ages = {}
        self.genders = {}

        def process_patient(patient):
            signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, 'raw', f'{patient}'))

            header = wfdb.rdheader(os.path.join(self.data_folder, 'raw', f'{patient}'))
            annotations = wfdb.rdann(os.path.join(self.data_folder + 'raw', f'{patient}'), 'atr')

            age = int(header.comments[0].split(' ')[0].strip())
            gender = header.comments[1].split(' ')[0].strip()
            gender = 1 if gender == 'M' else 0 if gender == 'F' else - 1

            # r_peaks are in the original frequency, we will convert them later to the model's frequency after resampling the signal
            # we also convert the annotation symbol to the main class label and filter only valid annotations
            r_peaks = [(r_peak, convert_label(annotations.symbol[i])) for i, r_peak in enumerate(annotations.sample) if annotations.symbol[i] in valid_annotations]

            # filter classes when num_classes is 3 (not used for BenchECG)
            if self.num_classes == 3:
                r_peaks = [(r_peak, label) for r_peak, label in r_peaks if label in ['N', 'S', 'V']]

            return patient, signal, header, annotations, r_peaks, age, gender

        results = Parallel(n_jobs=-1)(delayed(process_patient)(patient) for patient in self.patients)
        # results = [process_patient(patient) for patient in self.patients]

        ## RESAMPLING SIGNALS and R PEAKS
        for patient, signal, header, annotations, r_peaks, age, gender in results:
            if self.sampling_freq != header.fs:
                signal = nk.signal_resample(signal, sampling_rate=header.fs, desired_sampling_rate=self.sampling_freq, method='FFT')
                self.r_peaks[patient] = [(int(np.round(r_peak * self.freq_factor)), label, r_peak) for r_peak, label in r_peaks]
            else:
                self.r_peaks[patient] = [(r_peak, label, r_peak) for r_peak, label in r_peaks]

            self.signals[patient] = signal
            self.headers[patient] = header
            self.ages[patient] = age
            self.genders[patient] = gender


    def load_samples(self, subset):
        def process_sample(patient, r_peaks):
            samples = []
            last_class = None
            skipped = 0
            len_signal = len(self.signals[patient])

            if self.r_peaks_detection:
                # len signal and win_len are in the same frequency domain (resampled - 100)
                for i in range(0, len_signal, self.win_len * 2):
                    # current_r_peaks = [r for r, _, _ in r_peaks if i <= r < i + self.win_len * 2]
                    
                    # save both original and resampled
                    current_r_peaks = [
                        (r_tgt, r_orig) for r_tgt, _, r_orig in r_peaks 
                        if i // self.freq_factor <= r_orig < (i + self.win_len * 2) // self.freq_factor
                    ]

                    samples.append({
                        'start': i,
                        'end': min(i + self.win_len * 2, len_signal),
                        'patient': patient,
                        'r_peak': -1,
                        'around_r_peaks': current_r_peaks,
                    })
            elif subset == 'train':
                if self.skip_majority_class_samples:
                    for i, r_peak in enumerate(r_peaks):
                        # 0 is the position, 1 is the label
                        actual_class = r_peaks[i][1]

                        if (actual_class != last_class or skipped > 10 or actual_class != 'N'):
                            samples.append({
                                'patient': patient,
                                'start': max(0, r_peak[0] - self.win_len - self.context_len),
                                'end': min(r_peak[0] + self.win_len + self.context_len, len_signal),
                                # 'r_peak': r_peak[0],
                                'around_r_peaks': [(r, l) for r, l, _ in r_peaks if r_peak[0] - self.win_len + 1 <= r < r_peak[0] + self.win_len - 1],
                            })
                            skipped = 0
                        else:
                            skipped += 1
                        last_class = actual_class

                else:
                    for i, r_peak in enumerate(r_peaks):
                        samples.append({
                            'patient': patient,
                            'start': max(0, r_peak[0] - self.win_len - self.context_len),
                            'end': min(r_peak[0] + self.win_len + self.context_len, len_signal),
                            'around_r_peaks': [(r, l) for r, l, _ in r_peaks if r_peak[0] - self.win_len + 1 <= r < r_peak[0] + self.win_len - 1],
                        })
            else:
                #print(f"sample_len: {sample_len}, win_len: {self.win_len}, freq_factor: {self.freq_factor}")
                for i in range(0, len_signal, self.win_len * 2):
                    # for every window
                    around_r_peaks = [(r, l) for r, l, _ in r_peaks if i <= r < i + self.win_len * 2 - 1]
                    # print(f"i: {i}, win_len: {self.win_len}, sample_len: {sample_len}, around_r_peaks: {around_r_peaks}")
                    samples.append({
                        'patient': patient,
                        'start': max(0, i - self.context_len),
                        'end': min(i + self.win_len * 2 + self.context_len, len_signal),
                        'around_r_peaks': around_r_peaks,
                    })

            return samples

        results = Parallel(n_jobs=-1)(
            delayed(process_sample)(patient, self.r_peaks[patient]) for patient in self.patients
        )

        # Flatten results and reindex with unique keys
        self.samples = {i: sample for i, sample in enumerate(sum(results, []))}
        
    def __len__(self):
        return len(self.samples)
    
    def get_label_int(self, label):
        if label == 'N': return 0
        if label == 'S': return 1
        if label == 'V': return 2
        if label == 'F': return 3
        if label == 'Q': return 4
        else: raise ValueError(f'Unknown label {label}')

    def __getitem__(self, idx):
        if self.r_peaks_detection:
            return self.get_item_r_peaks(idx)
        else:
            return self.get_item_classification(idx)
        

    def get_item_r_peaks(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        header = self.headers[patient]
        start = sample['start']
        end = sample['end']
        signal = self.signals[patient][start:end]

        r_peaks_mask = np.zeros(signal.shape[0], dtype=np.float32)

        if len(sample['around_r_peaks']) > 0:
            indexes = np.array([r_tgt for r_tgt, r_orig in sample['around_r_peaks']]) - start
            r_peaks_mask[indexes] = 1

        # indexes = np.round(np.array(sample['around_r_peaks'])).astype(int) - start
        # print(f"indexes: {indexes}, start: {start}, freq_factor: {self.freq_factor}")
        # r_peaks_mask[indexes] = 1

        signal = self.filter_leads(signal, header.__dict__['sig_name'])
        if self.augmentations is not None:
            signal = self.augmentations(signal)

        orig_start = np.round(start / self.freq_factor)
        around_r_peaks = np.array([r_orig - orig_start for _, r_orig in sample['around_r_peaks']])

        # around_r_peaks = np.array([np.round((r - start) / self.freq_factor) for r in sample['around_r_peaks']])

        return {
            'signals': signal,
            'patient_id': patient,
            'r_peak': r_peaks_mask,
            'r_peak_orig': around_r_peaks
        }

    def get_item_classification(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        signal = self.signals[patient]
        header = self.headers[patient]
        age = self.ages[patient]
        gender = self.genders[patient]

        len_signal = signal.shape[0]

        window_start = sample.get('start', 0)
        window_end = sample.get('end', len_signal)

        # random shift is a percentage of context_len (let's say max 25% of it)
        if self.context_len > 0 and self.random_shift:
            random_shift = random.randint(0, self.patch_size // 2) - self.patch_size // 2
            if window_start + random_shift >= 0 and window_end - random_shift <= len_signal:
                window_start = window_start + random_shift
                window_end = window_end + random_shift

        window_signal = signal[window_start:window_end]
        sig_len = window_end - window_start
        window_signal = self.filter_leads(window_signal, header.__dict__['sig_name'])

        if self.augmentations is not None:
            window_signal = self.augmentations(window_signal)

        labels_mask = np.full(window_signal.shape[0], -1, dtype=np.int64)
        # labels_mask = np.zeros(window_signal.shape[0], dtype=np.long) - 1

        for r, l in sample['around_r_peaks']:
            if window_start <= r < window_end:
                labels_mask[r - window_start] = self.get_label_int(l)
                # if the r_peak is at the very beginning or very end of a patch, add a label for previous or next (only in training)
                if self.split == 'train' and self.extend_labels:
                    position = (r - window_start) % self.patch_size
                    # this is when is at the very beginning
                    if position == 0:
                        labels_mask[max(0, r - window_start - 1)] = self.get_label_int(l)
                    elif position == 1 and self.patch_size > 2:
                        labels_mask[max(0, r - window_start - 2)] = self.get_label_int(l)
                    elif position == 2 and self.patch_size > 3:
                        labels_mask[max(0, r - window_start - 3)] = self.get_label_int(l)
                    # this is when it is at the very end
                    elif position == self.patch_size - 1:
                        labels_mask[min(sig_len - 1, r - window_start + 1)] = self.get_label_int(l)
                    elif position == self.patch_size - 2 and self.patch_size > 2:
                        labels_mask[min(sig_len - 1, r - window_start + 2)] = self.get_label_int(l)
                    elif position == self.patch_size - 3 and self.patch_size > 3:
                        labels_mask[min(sig_len - 1, r - window_start + 3)] = self.get_label_int(l)


        return {
            'signal': window_signal,
            'patient_id': patient,
            'label': labels_mask,
            'age': age,
            'gender': gender,
            # 'r_peak_orig': original_r_peaks,
        }

    def filter_leads(self, signal, leads):
        # convert leads if needed 
        for i, lead in enumerate(leads):
            if lead in conversion.keys():
                leads[i] = conversion[lead]

        # print('leads', leads)

        signal_to_return = np.zeros((signal.shape[0], len(self.leads_to_use)), dtype=np.float32)
        # if the leads to use are not present in the signal set them to zero
        # leads present in the signal that are not in the leads to use are removed
        # the rest is kept unchanged
        for i, lead in enumerate(self.leads_to_use):
            if lead not in leads:
                signal_to_return[:, i] = np.zeros(signal.shape[0])
            else:
                signal_to_return[:, i] = signal[:, leads.index(lead)]
        return signal_to_return

class ECGMITBIHDatasetSingleHB(ECGMITBIHDataset):
    def __init__(self, config, split='train', augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        super().__init__(config, split, augmentations)

    @override
    def load_samples(self, subset):
        self.samples = []
        for patient in tqdm(self.patients, desc="Processing patients"):
            for i, (r_peak, _, _) in enumerate(self.r_peaks[patient]):
                if r_peak[0] > 0:
                    signal = self.signals[patient]
                    self.samples.append({
                        'patient': patient,
                        'r_peak': r_peak,
                        'signal': signal[max(0, r_peak[0] - 200): min(len(signal), r_peak[0] + 200)],
                    })

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        signal = sample['signal']
        label = sample['r_peak'][1]
        header = self.headers[patient]

        signal = self.filter_leads(signal, header.__dict__['sig_name'])

        if self.augmentations is not None:
            signal = self.augmentations(signal)


        return {
            'signal': signal,
            'patient_id': patient,
            'label': np.array([self.get_label_int(label)]),
        }


def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        signals = [torch.from_numpy(item['signals'].copy()).float() for item in batch]
        patients = [item['patient_id'] for item in batch]
        
        if 'r_peak' not in batch[0].keys(): 
            r_peaks = None
        else:
            r_peaks = [torch.from_numpy(item['r_peak']) for item in batch]
            r_peaks = torch.nn.utils.rnn.pad_sequence(r_peaks, batch_first=True)

        if 'label' not in batch[0].keys():
            labels = None
        else:
            labels = [torch.from_numpy(item['label']) for item in batch]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

            if not config.single_hb and not config.r_peaks_detection:
                labels = labels.unfold(1, config.patch_size, config.patch_size).max(dim=-1)[0].long()

        if 'r_peak_orig' not in batch[0].keys():
            r_peaks_orig = None
        else:
            r_peaks_orig = [torch.from_numpy(item['r_peak_orig']) for item in batch]
            r_peaks_orig = torch.nn.utils.rnn.pad_sequence(r_peaks_orig, batch_first=True, padding_value=torch.nan)

        # pad to same length and pad to match the patch size module
        if config.shuffle_baseline_wander_in_batch and split == 'train':
            signals = baseline_shuffler(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True))
        else:
            signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
            

        return {
            'signals': signals,
            'patient_ids': torch.tensor(patients),
            'r_peak': r_peaks,
            'labels': labels,
            'r_peak_orig': r_peaks_orig,
            # 'ages': torch.tensor([item['age'] for item in batch], dtype=torch.float32),
            #'genders': torch.tensor([item['gender'] for item in batch], dtype=torch.float32),
        }

    return collate_fn
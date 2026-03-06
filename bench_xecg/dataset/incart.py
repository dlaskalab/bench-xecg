
import os
from joblib import Parallel, delayed
from typing_extensions import override

from wandb.util import np
import wfdb

from .pretraining_dataset import PretrainDataset

class ECGIncartDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        
        self.data_folder = config.data_folder_incart
        self.win_len = config.win_len_incart

        self.load_records()
        self.load_samples()

    def load_records(self):
        with open(os.path.join(self.data_folder, 'RECORDS'), 'r') as f:
            self.records = [line.strip()[:3] for line in f.readlines()]
            print(f'INCART: loaded {len(self.records)}')

    def load_samples(self):
        self.samples = []

        def process_patient(patient):
            signal, header = wfdb.rdsamp(os.path.join(self.data_folder, f'{patient}'))

            signal = self.resample_if_needed(signal, header)

            samples = []
            for start in range(0, len(signal), self.win_len * self.sampling_freq):
                end = start + self.win_len * self.sampling_freq
                # if the segment is long enough
                if end <= len(signal) and (end - start) >= self.sampling_freq * 5: 
                    segment = signal[start:end, :]
                    samples.append((segment, header))

            return samples

        results = Parallel(n_jobs=-1)(delayed(process_patient)(patient) for patient in self.records)

        for samples in results:
            self.samples.extend(samples)

        print(f'Incart: loaded {len(self.samples)} samples from {len(self.records)} records.')

    def read_comment(self, header, key):
        # for each record read the comments
        comments = header['comments'][0].split(' <')
        for c in comments:
            if key in c.lower():
                return c.split(':')[-1].strip()
        return np.nan


    @override
    def __len__(self):
        return len(self.samples)
        
    @override
    def __getitem__(self, idx):
        signal, info = self.samples[idx]
        
        try:
            age = int(self.read_comment(info, 'age'))
        except:
            age = np.nan

        gender = self.read_comment(info, 'sex')
        gender = 1 if gender.lower() == 'm' else 0 if gender.lower() == 'f' else np.nan
        
        self.map_leads_and_clean(signal, info)

        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(signal) for _ in range(self.n_global_view)]
        else:
            global_signals = signal
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(signal) for _ in range(self.n_local_view)]
        else:
            local_signals = None

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
            'global_ages': [age] * self.n_global_view,
            'global_genders': [gender] * self.n_global_view,
            'local_ages': [age] * self.n_local_view if local_signals is not None else None,
            'local_genders': [gender] * self.n_local_view if local_signals is not None else None,   
        }
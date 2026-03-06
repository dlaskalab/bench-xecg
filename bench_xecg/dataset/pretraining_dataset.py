import os

import wfdb
import neurokit2 as nk
import numpy as np
import torch
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, verbose=0)

leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
jepa_leads =  ['i', 'ii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

mappings = { 'di': 'i', 'dii': 'ii', 'diii': 'iii' }

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        self.leads = config.leads if not config.use_ecg_jepa else jepa_leads
        self.leads = [l.lower() for l in self.leads] # ensure leads are lowercase
        print('Using leads :', self.leads)
        self.patch_size = config.patch_size
        self.split = split
        self.global_augmentations = global_augmentations
        self.local_augmentations = local_augmentations
        self.n_global_view = config.n_global_view
        self.n_local_view = config.n_local_view
        self.sampling_freq = config.sampling_freq
        self.nk_clean = config.nk_clean        
        self.max_length_signal = config.max_length_signal


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])
        age = self.ages[idx]
        gender = self.genders[idx]

        s, info = wfdb.rdsamp(os.path.join(self.data_folder, record))

        # mapping leads in the correct position
        s = self.map_leads_and_clean(s, info)
        s = self.resample_if_needed(s, info)

        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(s).copy() for _ in range(self.n_global_view)]
        else:
            global_signals = [s]
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(s).copy() for _ in range(self.n_local_view)]
        else:
            local_signals = []

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
            'global_ages': [age] * self.n_global_view,
            'global_genders': [gender] * self.n_global_view,
            'local_ages': [age] * self.n_local_view,
            'local_genders': [gender] * self.n_local_view,
        }
    
    def map_leads_and_clean(self, signal, info):
        s = np.zeros((len(signal), len(self.leads)))

        for lead in info['sig_name']:
            l = lead.lower()
            if l in mappings:
                l = mappings[l]
            if l in self.leads:
                if self.nk_clean:
                    s[:, self.leads.index(l)] = nk.ecg_clean(signal[:, info['sig_name'].index(lead)], sampling_rate=info['fs']).copy()
                else:
                    s[:, self.leads.index(l)] = signal[:, info['sig_name'].index(lead)]

        return s
    
    def resample_if_needed(self, signal, info):
        if self.sampling_freq != info['fs']:
            signal = nk.signal_resample(signal, sampling_rate=info['fs'], desired_sampling_rate=self.sampling_freq, method='FFT')   
            
        return signal
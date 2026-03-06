import os
from pathlib import Path

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split

from .pretraining_dataset import PretrainDataset


class ECGCODE15Dataset(PretrainDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = Path(config.data_folder_code15)
        self.labels_file = Path(config.labels_file_code15)
        self.load_tabular_data()
        self.load_records()
        self.load_age_gender()

    def load_records(self):
        self.records = self.tab_data.index.tolist()
        print(f'sample path CODE15: {self.records[0]}')
        print(f'loaded {len(self.records)} records')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        print("tabular data fields for CODE 15: ", self.tab_data.head())

        self.tab_data['valid'] = self.tab_data.parallel_apply(lambda row: (self.data_folder / f"{row['exam_id']}.hea").exists(), axis=1)
        self.tab_data = self.tab_data[self.tab_data['valid']]

        self.tab_data.set_index('exam_id', inplace=True)

    def load_age_gender(self):
        self.ages = self.tab_data['age'].tolist()
        self.genders = self.tab_data['is_male'].tolist()


class ECGCODE15AgeDataset(ECGCODE15Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        obj['age'] = self.tab_data.loc[self.records[idx], 'age']
        return obj
    
class ECGCODE15MortalityDataset(ECGCODE15Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        # drop rows that has nan in timey or death and count how many dropped
        original_count = self.tab_data.shape[0]
        self.tab_data = self.tab_data.dropna(subset=['timey', 'death'])
        print(f'dropped {original_count - self.tab_data.shape[0]} rows')
        self.records = self.tab_data.index.tolist()

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        return {
            'signal': obj['global_signals'][0],
            'death': torch.tensor(self.tab_data.loc[self.records[idx], 'death'], dtype=torch.bool),
            'timey': torch.tensor(self.tab_data.loc[self.records[idx], 'timey'], dtype=torch.float32)
        }

class ECGCODEDataset(PretrainDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = Path(config.data_folder_code)
        self.labels_file = Path(config.labels_file_code)
        self.annotation_file = Path(config.annotation_file_code)
        self.use_single_ecg = config.use_single_ecg
        self.load_tabular_data()

    def __len__(self):
        return len(self.unique_patients)
    
    def load_tabular_data(self):
        print("Loading tabular data...")
        
        df = pd.read_csv(self.labels_file)
        annotation_data = pd.read_csv(self.annotation_file)

        # We use regex to extract digits between TNMG and _
        df['id_exam'] = df['file_name'].str.extract(r'TNMG(\d+)_')[0].astype(int)
        df = df.merge(annotation_data, on='id_exam', how='left')

        print("Code dataframe: \n", df.head())
        
        # Group by patient ID
        grouped = df.groupby('id_patient')
        
        # Create a list where index i corresponds to self.unique_patients[i]
        self.patient_records = []
        
        # We iterate once through groups to build the fast lookup structure
        for _, group in grouped:
            # Store only what is needed as numpy arrays or lists
            record = {
                'file_names': group['file_name'].values,
                'ages': group['age'].values,
                # Assuming sex is constant per patient, take first
                'sex': 1 if group['sex'].iloc[0] == 'M' else 0 if group['sex'].iloc[0] == 'F' else np.nan
            }
            self.patient_records.append(record)

        # This list aligns with self.patient_records indices
        self.unique_patients = list(grouped.groups.keys())
        
        # Clean up heavy dataframe to free memory
        del df
        del annotation_data

        print(f"Data loaded. Found {len(self.patient_records)} unique patients.")


    def __getitem__(self, idx): 
        patient_data = self.patient_records[idx]
        
        files = patient_data['file_names']
        ages = patient_data['ages']
        gender = patient_data['sex']
                  
        num_views = self.n_global_view if not self.use_single_ecg else 1
        total_files = len(files)

        # Logic to pick indices
        if total_files > num_views:
            # Fast numpy choice without replacement
            selected_indices = np.random.choice(total_files, num_views, replace=False)
        else:
            # If we need more views than files available, we might need to duplicate or take all
            # Current logic: take all, then handle indexing below
            selected_indices = np.arange(total_files)

        selected_files = files[selected_indices]
        selected_ages = ages[selected_indices]

        # Load signals (I/O Bottleneck - see note below)
        # Using a set/dict comprehension to ensure we don't load the same file twice 
        # if the patient has duplicate file entries for some reason
        unique_file_names = set(selected_files)
        loaded_signals = { 
            rec: wfdb.rdsamp(str(self.data_folder / rec)) 
            for rec in unique_file_names 
        }
        
        signals_raw = [loaded_signals[f] for f in selected_files]

        # --- The rest of your processing logic remains mostly the same ---
        new_signals = []
        for signal_tuple in signals_raw:
            s, info = signal_tuple
            s = self.map_leads_and_clean(s, info)
            s = self.resample_if_needed(s, info)
            new_signals.append(s)
        
        signals = new_signals
    
        if self.global_augmentations is not None:
            global_signals = [self.global_augmentations(signals[i % len(signals)]) for i in range(self.n_global_view)]
            global_ages = [selected_ages[i % len(selected_ages)] for i in range(self.n_global_view)]
        else:
            global_signals = signals
            global_ages = selected_ages
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [self.local_augmentations(signals[(i + self.n_global_view) % len(signals)]) for i in range(self.n_local_view)]
            local_ages = [selected_ages[(i + self.n_global_view) % len(selected_ages)] for i in range(self.n_local_view)]
        else:
            local_signals = []
            local_ages = None

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
            'global_ages': global_ages,
            'global_genders': [gender] * self.n_global_view,
            'local_ages': local_ages,
            'local_genders': [gender] * self.n_local_view,
        }       
    


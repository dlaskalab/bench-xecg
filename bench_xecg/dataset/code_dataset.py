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
        self.load_records()

    def load_records(self):
        self.records = self.tab_data["file_name"].values
        print(f'CODE: sample path: {self.records[0]}')
        print(f'CODE: loaded {len(self.records)} records')
        print(f'CODE: number of unique patients {len(self.unique_patients)}')

    def __len__(self):
        return len(self.unique_patients)

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        self.annotation_data = pd.read_csv(self.annotation_file)
        # set exam_id as index
        # remove trace_file, patient_id and nn_predicted_age
        print("tabular data fields for CODE: ", self.tab_data.head())
        print("annotation data fields for CODE: ", self.annotation_data.head())

        self.tab_data['exam_id_sliced'] = self.tab_data.parallel_apply(lambda row: row['file_name'].split('/')[1].split('TNMG')[-1], axis=1)
        self.tab_data['exam_id'] = self.tab_data.parallel_apply(lambda row: row['exam_id_sliced'].split('_')[0], axis=1)

        # merge the annotation data to the tabular data on exam_id
        self.tab_data = self.tab_data.merge(self.annotation_data, on='exam_id', how='left')
        self.unique_patients = self.tab_data['patient_id'].unique()

        # set index patient id for faster retrieval
        self.tab_data = self.tab_data.set_index('patient_id').sort_index()

        print(f'Merged annotation data for CODE: ', self.tab_data.head())

    def __getitem__(self, idx):           
        patient = self.unique_patients[idx]

        # get all the rows with specifiefied patient id and get the corresponding records
        patient_rows = self.tab_data.loc[[patient]]
        
        # records = self.tab_data[self.tab_data['patient_id'] == int(patient)]['file_name'].tolist()
        num_views = self.n_global_view if not self.use_single_ecg else 1

        # Correct sampling
        if len(patient_rows) > num_views:
            patient_rows = patient_rows.sample(n=num_views)
            
        unique_records = patient_rows['exam_id_sliced'].unique()
        unique_signals = { rec: wfdb.rdsamp(str(self.data_folder / rec)) for rec in unique_records }
        signals = [ unique_signals[row.exam_id_sliced] for _, row in patient_rows.iterrows() ]

        ages = patient_rows['age'].values
        first_gender = patient_rows['sex'].iloc[0]
        gender = 1 if first_gender == 'M' else 0 if first_gender == 'F' else np.nan

        # mapping leads in the correct position
        new_signals = []
        for signal in signals:
            s, info = signal
            s = self.map_leads_and_clean(s, info)
            s = self.resample_if_needed(s, info)
            new_signals.append(s)
        signals = new_signals
    
        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(signals[i % len(signals)]) for i in range(self.n_global_view)]
            global_ages = [ages[i % len(ages)] for i in range(self.n_global_view)]
        else:
            global_signals = signals
            global_ages = ages
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(signals[(i + self.n_global_view)% len(signals)]) for i in range(self.n_local_view)]
            local_ages = [ages[(i + self.n_global_view) % len(ages)] for i in range(self.n_local_view)]
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
    


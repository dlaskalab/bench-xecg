import os

import torch
import wfdb
import pandas as pd

from .pretraining_dataset import PretrainDataset
from .generic_utils import pad, RandomSwitchBaselineWanderBatched

def extract_diagnosis_code_path(file_name):
    record = wfdb.rdheader(file_name)
    return extract_diagnosis_code_header(record)
        
def extract_diagnosis_code_header(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]
    return None

class ECGCPSC2018Dataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            config: configuration object
            split (str): 'train', 'val', 'test' or 'all'
            global_augmentations (list): list of global augmentations to apply
            local_augmentations (list): list of local augmentations to apply
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_cpsc2018

        self.load_records()
        self.load_labels()  
    
    def load_records(self):
        # loop over the folders in self.data_folder
        all_dirs = sorted([d for d in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, d))])
        exams = pd.DataFrame()
        all_files = []
        for d in all_dirs:
            with open(os.path.join(self.data_folder, d, 'RECORDS'), 'r') as f:
                lines = f.readlines()
            all_files += [os.path.join(d, line.strip()) for line in lines]

        print('CPSC2018 all files: ', len(all_files))
        
        exams['file_name'] = all_files
        # check all files exists
        exams['valid'] = exams.parallel_apply(lambda row: os.path.exists(os.path.join(self.data_folder, f"{row['file_name']}.hea")), axis=1)
        exams = exams[exams['valid']]
        exams['diagnosis_code'] = exams.parallel_apply(lambda row: extract_diagnosis_code_path(os.path.join(self.data_folder,row['file_name'])), axis=1)

        self.tab_data = exams
        print('CPSC2018 num ecgs: ', len(exams))

        self.records = self.tab_data['file_name'].tolist()
        if self.split == 'train':
            self.records = [record for record in self.records if 'g7' not in record and 'g6' not in record]
        elif self.split == 'val':
            self.records = [record for record in self.records if 'g6' in record]
        elif self.split == 'test':
            self.records = [record for record in self.records if 'g7' in record]
        elif self.split == 'all':
            pass
        
        self.tab_data = self.tab_data[self.tab_data['file_name'].isin(self.records)]
        print('CPSC2018:', self.tab_data.head())
        # print('CPSC label distribution: \n', self.tab_data['diagnosis_code'].value_counts())


    def load_labels(self):
        labels = self.tab_data['diagnosis_code'].unique()
        splitted_labels = []
        for label in labels:
            splitted_labels.extend(label.split(','))

        self.labels_unique = sorted(list(set(splitted_labels)))
    
    def __getitem__(self, idx):
        signal = super().__getitem__(idx)
        info = wfdb.rdheader(os.path.join(self.data_folder, self.records[idx]))

        label_str = extract_diagnosis_code_header(info).split(',')
        labels = [1 if label in label_str else 0 for label in self.labels_unique]

        signal = signal['global_signals'][0]  # Assuming global_signals is a list of signals

        return {
            'signal': signal,
            'labels':labels
        }


def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchBaselineWanderBatched(config.sampling_freq, 0.5)

    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        class_labels = [torch.tensor(item['labels']).float() for item in batch]

        # pad the signals to the same length
        if split == 'train' and config.shuffle_baseline_wander_in_batch:
            signals = baseline_shuffler(pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig) for sig in signals], batch_first=True).float(), patch_size=config.patch_size))
        else:
            signals = pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig.copy()) for sig in signals], batch_first=True).float(), patch_size=config.patch_size)
            
        return {
            'signals': signals,
            'class_labels': torch.stack(class_labels),
        }
    return collate_fn


class ECGCPSC2018AgeDataset(ECGCPSC2018Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.filter_valid_records()

    def filter_valid_records(self):
        # filter records that have age in the comments
        valid_records = []
        for record in self.records:
            info = wfdb.rdheader(os.path.join(self.data_folder, record))
            age = None
            for comment in info.comments:
                if comment.startswith('Age:'):
                    age = comment.split(': ')[1]
                    try:
                        int_age = int(age)
                        if int_age < 0 or int_age > 120:
                            print(f'Invalid age {age} in record {record}')
                        else:
                            valid_records.append(record)
                    except:
                        print(f'Invalid age {age} in record {record}')
                    break
        self.records = valid_records
        print(f'Filtered to {len(self.records)} records with age information')

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        info = wfdb.rdheader(os.path.join(self.data_folder, self.records[idx]))
        info.comments
        # regex 'Age: (\d+)' to extract age from comments
        age = None
        for comment in info.comments:
            if comment.startswith('Age:'):
                age = int(comment.split(': ')[1])
                break
        if age is None:
            raise ValueError(f'Age not found in comments for record {self.records[idx]}')

        obj['age'] = torch.tensor(age, dtype=torch.float32)
        return obj


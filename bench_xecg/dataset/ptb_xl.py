import os

import torch
import pandas as pd
import ast

from .pretraining_dataset import PretrainDataset
from .generic_utils import pad

class ECGPTBXLDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_ptbxl
        self.labels_file = os.path.join(config.data_folder_ptbxl, '..', 'ptbxl_database.csv')
        self.task = config.task
        self.classes = config.classes if not None else ['STTC', 'NORM', 'MI', 'HYP', 'CD']
        self.load_tabular_data()
        self.load_records(split, task=config.task)

    def load_records(self, split, task='multilabel'):
        # fold 19 is for testing, while fold 18 is for validation
        if split == 'train':
            # get all the tab data index where the fold is not 18 or 19
            self.tab_data = self.tab_data[self.tab_data['strat_fold'] != 9][self.tab_data['strat_fold'] != 10]
        elif split == 'val':
            # get all the tab data index where the fold is 18
            self.tab_data = self.tab_data[self.tab_data['strat_fold'] == 9]
        elif split == 'test':
            # get all the tab data index where the fold is 19
            self.tab_data = self.tab_data[self.tab_data['strat_fold'] == 10]

        if task == 'multiclass':
            self.tab_data['num_labels'] = self.tab_data.T.parallel_apply(lambda row: sum([1 if label in row['diagnostic_superclass'] else 0 for label in self.classes]))
            # keep only the records with sum == 1
            self.tab_data = self.tab_data[self.tab_data['num_labels'] == 1]

        if self.classes is not None:
            self.tab_data = self.tab_data[
                self.tab_data['diagnostic_superclass'].apply(
                    lambda x: any(item in self.classes for item in x)
                )
            ]
        else: 
            self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        self.records = self.get_records()
        self.ages = self.tab_data['age'].values.tolist()
        self.genders = self.tab_data['sex'].values.tolist()

        print("Tabular data fields for PTB-XL: \n", self.tab_data.head())
        # for each label count the number of occurrences
        table_count = self.tab_data['diagnostic_superclass'].explode().value_counts()
        print("Number of occurrences for each label in PTB-XL: ")
        print(table_count)

    def get_records(self):
        records = self.tab_data['filename_hr'].values.tolist()
        return [os.path.join(self.data_folder, record.split('/')[1], record.split('/')[2]) for record in records]

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file, index_col='ecg_id')
        self.tab_data.scp_codes = self.tab_data.scp_codes.apply(lambda x: ast.literal_eval(x))

        statements = pd.read_csv(os.path.join(self.data_folder, '../scp_statements.csv'), index_col=0)
        statements = statements[statements.diagnostic == 1]

        def aggregate_diagnostic(y_dic, statements=statements, column='diagnostic_class'):
            """
            Aggregate the diagnostic classes into superclasses and subclasses
            """
            tmp = []
            for key in y_dic.keys():
                if key in statements.index:
                    tmp.append(statements.loc[key].diagnostic_class)
            return list(set(tmp))
        
        self.tab_data['diagnostic_superclass'] = self.tab_data.scp_codes.apply(lambda x: aggregate_diagnostic(x, statements=statements, column='diagnostic_superclass'))
        self.tab_data['diagnostic_subclass'] = self.tab_data.scp_codes.apply(lambda x: aggregate_diagnostic(x, statements=statements, column='diagnostic_subclass'))

        print("Classes for PTB-XL: ", self.classes)
        self.subclasses = ['STTC', 'NST_', 'NORM', 'IMI', 'AMI', 'LVH', 'LAFB/LPFB', 'ISC_', 'IRBBB', '_AVB', 'IVCD', 'ISCA', 'CRBBB', 'CLBBB', 'LAO/LAE', 'ISCI', 'LMI', 'RVH', 'RAO/RAE', 'WPW', 'ILBBB', 'SEHYP', 'PMI'] # statements['diagnostic_subclass'].unique()
        print("Subclasses for PTB-XL: ", self.subclasses)

        # change type of age columns from float to int
        self.tab_data['age'] = self.tab_data['age'].fillna(0)
        self.tab_data['age'] = self.tab_data['age'].astype(int)
        # remove some unised columns

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.tab_data.iloc[idx]
        superclass_label = row['diagnostic_superclass']
        # convert the superclass label to a one-hot encoding
        superclass_label = [1 if label in superclass_label else 0 for label in self.classes]

        subclass_label = row['diagnostic_subclass']
        # convert the subclass label to a one-hot encoding
        subclass_label = [1 if label in subclass_label else 0 for label in self.subclasses]
  
        class_info = {
            'class_label': torch.tensor(superclass_label, dtype=torch.float32),
            'subclass_label': torch.tensor(subclass_label, dtype=torch.float32),
        }

        signal = super().__getitem__(idx)
        # concatenate the two dictionaries
        signal.update(class_info)
        return signal

class ECGPTBXLAgeDataset(ECGPTBXLDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split, global_augmentations, local_augmentations)
        # remove data with missing age information
        self.tab_data = self.tab_data[self.tab_data['age'].notna() & (self.tab_data['age'] < 100)]

        # refresh the records after changing tab data
        self.records = self.get_records()


    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        obj['age'] = torch.tensor(self.tab_data.iloc[idx]['age'], dtype=torch.float32)
        return obj


def make_collate_fn(config):
    def collate_fn(batch):
        signals = [item['global_signals'][0] for item in batch]

        superclass_labels = [item['class_label'] for item in batch]
        subclass_labels = [item['subclass_label'] for item in batch]

        signals = pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig.copy()) for sig in signals], batch_first=True).float(), patch_size=config.patch_size)
            
        tortn = {
            'signals': signals,
            'class_labels': torch.stack(superclass_labels),
            'subclass_labels': torch.stack(subclass_labels),
        }

        if 'global_ages' in batch[0]:
            tortn['ages'] = torch.stack([torch.tensor(sample['global_ages'][0]) for sample in batch], dim=0)
        if 'global_genders' in batch[0]:
            tortn['genders'] = torch.stack([torch.tensor(sample['global_genders'][0]) for sample in batch], dim=0)

        if batch[0].get('age') is not None:
            tortn['age'] = torch.stack([item['age'] for item in batch])

        return tortn

    return collate_fn

    
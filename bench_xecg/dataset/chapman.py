from pathlib import Path

import wfdb
import pandas as pd
import numpy as np

from .pretraining_dataset import PretrainDataset

class ECGChapmanDataset(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = Path(config.data_folder_chapman)
        self.labels_file = Path(config.labels_file_chapman)

        self.age_gender_labels_file = self.data_folder / 'age_gender_labels.csv'
    
        self.load_tabular_data()
        self.load_records(split)
        self.load_age_gender()
        
    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        self.records = sorted(self.tab_data['file_name'].tolist())

    def load_age_gender(self):
        # if the age and gender file is empty or does not exist, we will read the age and gender from the comments of the records
        if not self.age_gender_labels_file.exists():
            print("Age and gender file does not exist, reading from comments...")
            # for each record read the comments
            self.ages = []
            self.genders = []
            for record in self.records:
                header = wfdb.rdheader(self.data_folder / record)
                comment = header.__dict__['comments']

                # add age
                age_added = False
                for c in comment:
                    if 'age' in c.lower():
                        try:
                            if int(c[4:].strip()) < 0:
                                continue
                            self.ages.append(int(c[4:].strip()))
                            age_added = True
                        except ValueError:
                            continue
                if not age_added:
                    self.ages.append(np.nan) # if age is not found, set it to nan

                # add gender
                sex_added = False
                for c in comment:
                    if 'sex' in c.lower():
                        if 'Male' in c:
                            self.genders.append(1)
                            sex_added = True
                        elif 'Female' in c:
                            self.genders.append(0)
                            sex_added = True
                if not sex_added:
                    self.genders.append(np.nan)
            # save the age and gender to a csv file
            age_gender_df = pd.DataFrame({'file_name': self.records, 'age': self.ages, 'gender': self.genders})

            try:
                age_gender_df.to_csv(self.age_gender_labels_file, index=False)
                print(f"Saved age and gender information to {self.age_gender_labels_file}")
            except PermissionError:
                print(f"Permission error: Unable to save age and gender information to {self.age_gender_labels_file}. Please check your file permissions.")
            except Exception as e:
                print(f"Error saving age and gender information: {e}")
        else:
            # load age and gender from the existing file
            print("Age and gender file exists, loading from file...")
            age_gender_df = pd.read_csv(self.age_gender_labels_file)
            self.ages = age_gender_df['age'].tolist()
            self.genders = age_gender_df['gender'].tolist()
            self.records = age_gender_df['file_name'].tolist()


    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        print("Chapman&Ningbo - tabular data: ", self.tab_data.head())
        print(f'Chapman&Ningbo - colums {self.tab_data.columns}')
        print(f'Chapman&Ningbo - number of samples: {len(self.tab_data)}')

    def __len__(self):
        return len(self.records)

         
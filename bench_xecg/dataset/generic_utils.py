import os 
from joblib import Parallel, delayed

import torch
from torchvision import transforms
from ..augmentations import *

def get_transforms(config, split='train', type=None):
    """
    """
    t = transforms.Compose([])
    if config.normalize:
        t.transforms.append(Normalize())

    if config.standardize:
        t.transforms.append(Standardize())

    if config.low_pass_filter:
        t.transforms.append(LowpassFilter(config.sampling_freq, config.low_pass_filter))
    
    if config.high_pass_filter:
        t.transforms.append(HighpassFilter(config.sampling_freq, config.high_pass_filter))
    
    if split != 'train': 
        t.transforms.append(CropFixedLen(config.max_length_signal))
        return t
    
    if config.random_paper_layout > 0.:
        t.transforms.append(ECGPaperLayoutMask(p=config.random_paper_layout, format_ratio=config.random_paper_6x2_prob ))


    if config.random_crop < 1. and config.random_crop > 0.:
        t.transforms.append(RandomCrop(
            config.global_random_crop if type == 'global' else  config.local_random_crop if type == 'local' else config.random_crop,
            max_length=config.max_length_signal
        ))
    else:
        t.transforms.append(CropFixedLen(config.max_length_signal))

    if config.shift_baseline_wander_in_sample:
        t.transforms.append(RandomShiftBaselineWander(config.sampling_freq, 0.5))

    if config.random_drop_leads > 0.:
        t.transforms.append(RandomDropLeads(config.random_drop_leads, keep_lead_II = config.keep_lead_II))

    if config.random_surrogate_prob > 0.:
        t.transforms.append(FTSurrogate(0.05, prob=config.random_surrogate_prob))
    if config.random_jitter_prob > 0.:  
        t.transforms.append(Jitter(sigma=0.1, prob=config.random_jitter_prob))
    if config.random_resample:
        t.transforms.append(RandomResample(config.sampling_freq, 0.03))
    
    if config.random_change_amplitude > 0.:
        t.transforms.append(RandomChangeAmplitude(amplitude_range=0.2, prob=config.random_change_amplitude))
    return t


def get_max_n_jobs(default=-1):
    n_jobs = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    return n_jobs

    
def find_records(folder, header_extension='.dat'):
    def process_file(root, file):
        extension = os.path.splitext(file)[1]
        if extension == header_extension:
            record = os.path.relpath(os.path.join(root, file), folder)[:-len(header_extension)]
            return record
        return None

    records = set()

    print(f'Finding records in {folder}...')
    results = Parallel(n_jobs=get_max_n_jobs())(delayed(process_file)(root, file) for root, _, files in os.walk(folder) for file in files)
    records.update(filter(None, results))
    records = sorted(records)
    return records



def make_collate_fn(config):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        # Pad and clean global signals
        result = {
            'global_signals': pad_multi_view_batch([sample['global_signals'] for sample in batch], config.patch_size),
        }

        if config.shuffle_baseline_wander_in_batch:
            # Apply baseline shuffling to the global signals
            result['global_signals'] = torch.stack([
                baseline_shuffler(result['global_signals'][view, ...]) 
                for view in range(result['global_signals'].shape[0])
            ], dim=0)

            # print(f'Applied baseline shuffling to global signals with shape: {result["global_signals"].shape}')

        # Optional: handle local signals if present
        if 'local_signals' in batch[0] and batch[0]['local_signals'] is not None:
            result['local_signals'] = pad_multi_view_batch([sample['local_signals'] for sample in batch], config.patch_size)
            
            if config.shuffle_baseline_wander_in_batch:
                # Apply baseline shuffling to the local signals
                result['local_signals'] = torch.stack([
                    baseline_shuffler(result['local_signals'][view, ...]) 
                    for view in range(result['local_signals'].shape[0])
                ], dim=0)

        if 'global_ages' in batch[0]:
            result['global_ages'] = torch.stack([torch.tensor(sample['global_ages']) for sample in batch], dim=1)
        if 'global_genders' in batch[0]:
            result['global_genders'] = torch.stack([torch.tensor(sample['global_genders']) for sample in batch], dim=1)
        if 'local_ages' in batch[0]:
            result['local_ages'] = torch.stack([torch.tensor(sample['local_ages']) for sample in batch], dim=1)
        if 'local_genders' in batch[0]:
            result['local_genders'] = torch.stack([torch.tensor(sample['local_genders']) for sample in batch], dim=1)

        return result

    return collate_fn


def make_collate_fn_task(config, key_label='age'):
    def collate_fn(batch):
        if 'signal' in batch[0]:
            # If 'signal' is present, use it
            signals = [item['signal'] for item in batch]
            signals = pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig.copy()) for sig in signals], batch_first=True).float(), patch_size=config.patch_size)
        else:
            signals = [item['global_signals'][0] for item in batch]
            signals = pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig.copy()) for sig in signals], batch_first=True).float(), patch_size=config.patch_size)

        if isinstance(key_label, list):
            return {
                'signals': signals,
                **{k: torch.stack([item[k] for item in batch]) for k in key_label}
            }
        elif key_label is not None:
            return {
                'signals': signals,
                key_label: torch.stack([
                    item[key_label].float() if isinstance(item[key_label], torch.Tensor)
                    else torch.tensor(item[key_label]).float()
                    for item in batch
                ])
            }
        return {
            'signals': signals
        }

    return collate_fn

def pad(x, patch_size):
    if x.dim() == 2:
        x = x.unsqueeze(-1)
        
    length = x.shape[1]
    excess = length % patch_size
    if excess != 0:
        x = x[:, :-excess, :]
    return x

def pad_multi_view_batch(sample_list, patch_size):
    """
    Pads a batch of multi-view signals to ensure each signal's length is a multiple of patch_size.
    Args:
        sample_list (list): Batched list of multi-view signals, where each element is a list of views
        patch_size (int): The patch size to pad to
    Returns:
        torch.Tensor: Padded batch of multi-view signals of shape (batch_size, n_views, seq_len, n_leads)
    """
    
    sample_list = [[sample[i] for sample in sample_list] for i in range(len(sample_list[0]))]

    topad = [ 
        pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig) for sig in signal], batch_first=True).float(), patch_size).permute(1, 0, 2)
        for signal in sample_list
    ]

    # print('topad shapes: ', [top.shape for top in topad])
    tortn =  torch.nn.utils.rnn.pad_sequence(topad, batch_first=True).permute(0, 2, 1, 3)

    # print(f'Padded multi-view batch to shape: {tortn.shape}')
    return tortn


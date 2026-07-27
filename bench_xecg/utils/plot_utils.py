from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

color_1 = (50 / 255, 134 / 255, 143 / 255)
color_2 = (207/ 255, 86/ 255, 86/ 255)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_reconstruction(sample, model, patch_size, freq, device, logdir, epoch, name, training_strategy):
    with torch.no_grad():

        x = torch.tensor(sample['global_signals'][0], dtype=torch.float32).to(device).unsqueeze(0)
        orig_signal = x.clone()
        x = F.pad(x, (0, 0, 0, patch_size - x.shape[1] % patch_size))

        out = model(x, masking=True, reconstruct=True)
        reconstruct = out['reconstruction']
        mask = out['mask']

        if training_strategy == 'next_token_prediction':
            orig_signal = orig_signal[:, :reconstruct.shape[1]]
            orig_signal = orig_signal[:, patch_size:].squeeze()
            reconstruct = reconstruct[:, :-patch_size]


        # try to reconstruct one element at a time
        reconstruct = reconstruct.view(1, -1, orig_signal.shape[-1])
        orig_signal = orig_signal.view(1, -1, orig_signal.shape[-1])

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(x.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(orig_signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(orig_signal[..., i].cpu().squeeze().numpy(), color=color_1)
            # print('shift_reconstruct shape', shift_reconstruct.shape)
            ax.plot(reconstruct[..., i].cpu().squeeze().numpy(), color=color_2)

            # sometimes the signal is zeroed out by the random drop leads
            # so here i just plot the zeroed out signal with a dashed line to know that channel was zeroed out
            #has_augmentation = False
            #if (x[..., i] != orig_signal[..., i]).any():
            #    ax.plot(x[..., patch_size:reconstruct.shape[1], i].cpu().squeeze().numpy(), color='grey', linestyle='--')
            #    has_augmentation = True
                
            ax.set_title(leads[i])

            # add vertical lines avery patch size
            for j in range(0, x.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)
                
            if training_strategy == 'sim_dino_v2' or training_strategy == 'lejepa_masked':
                if mask.shape[0] == 1:
                    ax_mask = mask.squeeze()
                else:
                    ax_mask = mask[i]

                # mask is at patch resolution; expand to sample resolution to match x's x-axis
                ax_mask = ax_mask.detach().cpu().repeat_interleave(patch_size)

                ax.fill_between(
                    list(range(x.shape[1])),
                    orig_signal[..., i].min().item(),
                    orig_signal[..., i].max().item(),
                    where=ax_mask,
                    color='red',
                    alpha=0.3,
                    label='mask'
                )
        

            # ax.set_yticks([])
            if i == 0:
                if training_strategy == 'sim_dino_v2' or training_strategy == 'lejepa_masked':
                    ax.legend(['Original', 'Reconstructed', 'Mask'], loc='upper left')
                else:
                    ax.legend(['Original', 'Reconstructed'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(x[0]), freq))
                ax.set_xticklabels(np.arange(0, len(x[0]), freq) // freq)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/reconstruction_{name}.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
    
def plot_local_views(sample, patch_size, freq, device, logdir, epoch, name):
    with torch.no_grad():
        num_signals = len(sample['local_signals'])
        leads = [f"Lead {i+1}" for i in range(sample['local_signals'][0].shape[-1])]
        color_1 = "tab:blue"

        fig = plt.figure(figsize=(20, 15))

        # garantisce almeno 1 riga
        n_rows = max(1, num_signals // 2)
        n_cols = 2 if num_signals > 1 else 1

        outer_gs = gridspec.GridSpec(n_rows, n_cols)
        outer_gs.update(wspace=0.2, hspace=0.3)

        for idx, signal in enumerate(sample['local_signals']):
            x = torch.from_numpy(signal).float().unsqueeze(0).to(device)

            # Calcolo corretto posizione nella griglia principale
            row = idx // n_cols
            col = idx % n_cols

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                6, 2, subplot_spec=outer_gs[row, col], wspace=0.2, hspace=0.4
            )

            for i in range(x.shape[-1]):
                ax = plt.subplot(inner_gs[i // 2, i % 2])
                ax.plot(x[..., i].cpu().squeeze().numpy(), color=color_1)
                ax.set_title(leads[i])

                for j in range(0, x.shape[1], patch_size):
                    ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

                if i == 10 or i == 11:
                    ax.set_xticks(np.arange(0, len(x[0]), freq))
                    ax.set_xticklabels(np.arange(0, len(x[0]), freq) // freq)
                    ax.set_xlabel('Time (s)')
                else:
                    ax.set_xticks([])

        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)
        path = f'{logdir}/epoch_{epoch}/reconstruction_{name}.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path



    
def plot_generation(sample, model, patch_size, device, logdir, epoch, name):
    with torch.no_grad():
        signal = sample['signal'].to(device).unsqueeze(0)

        signal = signal[:, :signal.shape[1] - signal.shape[1] % patch_size]
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(-1)

        generated = model.generate(signal, length=(2048 // patch_size))

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(signal.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(signal[..., i].cpu().squeeze().numpy(), color=color_1)
            ax.plot(
                range(signal.shape[1], signal.shape[1] + generated.shape[1]),
                generated[..., i].cpu().squeeze().numpy(), color=color_2)
            ax.set_title(leads[i],)

            # add vertical lines avery patch size
            for j in range(0, signal.shape[1] + generated.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # ax.set_yticks([])
            if i == 0:
                ax.legend(['Original', 'Generated'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(signal[0]), 360))
                ax.set_xticklabels(np.arange(0, len(signal[0]), 360) // 360)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/generation_{name}.png'
        plt.savefig(path)
        plt.close()
        return path

def plot_latent_space(z, epoch, logdir):
    """
    Performs SVD on the covariance matrix of embeddings to check for dimensional collapse.
    z: Tensor of shape [N_samples, Embedding_Dim]
    """
    # Move to CPU/Numpy for analysis (SVD on large matrix can be heavy on GPU)
    z = z.float()
    
    # 1. Center the embeddings
    z = z - z.mean(dim=0)
    
    # 2. Compute Covariance Matrix
    N, D = z.shape
    if N < 2: return # Not enough samples

    cov_matrix = (z.T @ z) / (N - 1)
    
    # 3. SVD
    # Compute singular values of the covariance matrix
    _, S, _ = torch.svd(cov_matrix)
    
    # 4. Calculate Effective Rank (Entropy)
    total_variance = torch.sum(S)
    p_vals = S / total_variance
    entropy = -torch.sum(p_vals * torch.log(p_vals + 1e-12))
    effective_rank = torch.exp(entropy)
    
    # Log scalar metrics
    # self.log('val_effective_rank', effective_rank.item(), prog_bar=True)
    # self.log('val_embedding_variance', total_variance.item(), prog_bar=False)

    # 5. Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Spectrum (Log scale)
    s_np = S.cpu().numpy()
    ax[0].plot(s_np, marker='.', markersize=2)
    ax[0].set_yscale('log')
    ax[0].set_title(f'Singular Value Spectrum (Eff Rank: {effective_rank:.1f}/{D})')
    ax[0].set_xlabel('Singular Value Index')
    ax[0].set_ylabel('Value')
    ax[0].grid(True, which="both", ls="-", alpha=0.2)

    # Plot Covariance Heatmap (First 50 dims)
    cov_np = cov_matrix.cpu().numpy()
    vis_dim = min(50, D)
    sns.heatmap(cov_np[:vis_dim, :vis_dim], ax=ax[1], cmap='viridis', center=0, vmin=-0.1, vmax=0.1, cbar=True)
    ax[1].set_title(f'Covariance (First {vis_dim} dims)')
    
    plt.tight_layout()

    os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)
    path = f'{logdir}/epoch_{epoch}/latent_space.png'
    plt.savefig(path)
    plt.close()

    return effective_rank.item(), total_variance.item(), path


    

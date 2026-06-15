import os
import glob

def find_checkpoint(ckpt_root: str, run_id: str, task='age') -> str:
    """Return the single .ckpt file inside train-<task>/<run_id>/checkpoints/."""
    pattern = os.path.join(ckpt_root, run_id, 'checkpoints', '*.ckpt')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for run '{run_id}' (pattern: {pattern})")
    if len(matches) > 1:
        # Prefer 'best' in the name, otherwise take the last modified file
        best = [m for m in matches if 'best' in os.path.basename(m).lower()]
        matches = best if best else sorted(matches, key=os.path.getmtime)
    return matches[0]
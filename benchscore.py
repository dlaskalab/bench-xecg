import os
import glob
import re
import pandas as pd
import numpy as np
import typer
from typing import Union, List


app = typer.Typer()

# ================= CONFIGURATION =================
# Map folders to the specific metric column name in the CSV
TASK_METRIC_MAP = {
    "train-age": [
        "mimic/test_rsmape/dataloader_idx_1", 
        "ptbxl/test_rsmape/dataloader_idx_0", 
        "cpsc/test_rsmape/dataloader_idx_2"
    ],
    "train-cpsc2018-multilabel": "test_auroc",
    "train-exercise-r-peak": "test_f1_20", 
    "train-lab": "test_auroc_avg",            
    "train-mitbih-5": "test_f1/mean",       
    "train-mitbih-r_peaks": "test_f1_20",  
    "train-ptbxl-diagnosis_superclass-multilabel": "test_auroc",
    "train-sleep-apnea_bis": "test_auc",
    "train-survival": "test_ci",   
    "train-deepbeat": "test_auroc"
}
# =================================================


def get_sorted_versions(exp_path: str) -> List[int]:
    """
    Uses glob to find all 'version_*' folders and returns integers 
    sorted descending (newest first).
    """
    # Use glob to match the pattern /path/to/exp/version_*
    search_pattern = os.path.join(exp_path, "version_*")
    folder_paths = glob.glob(search_pattern)
    
    versions = []
    version_regex = re.compile(r'version_(\d+)')

    for path in folder_paths:
        if os.path.isdir(path):
            folder_name = os.path.basename(path)
            match = version_regex.search(folder_name)
            if match:
                versions.append(int(match.group(1)))
    
    # Sort descending (e.g., 5, 4, 3, 2, 1)
    versions.sort(reverse=True)
    return versions

def extract_raw_score(file_path: str, metric_config: Union[str, List[str]]) -> float:
    """
    Reads CSV. 
    - If metric_config is a list: computes row-wise mean of those columns.
    - If metric_config is a string: reads that column.
    Returns the value from the LAST row.
    """
    try:
        df = pd.read_csv(file_path)
        
        # CASE A: List of columns (Average them first)
        if isinstance(metric_config, list):
            # Ensure all columns exist
            missing = [c for c in metric_config if c not in df.columns]
            if missing:
                return None
            
            # Compute mean across columns (axis=1) -> get last row
            combined_series = df[metric_config].mean(axis=1)
            if combined_series.empty: 
                return None
            return combined_series.iloc[-1]

        # CASE B: Single column
        else:
            if metric_config not in df.columns:
                return None
            valid_vals = df[metric_config].dropna()
            if valid_vals.empty:
                return None
            return valid_vals.iloc[-1]

    except Exception:
        return None

@app.command()
def main(
    base_folder: str = typer.Argument(..., help="Path to the logs folder"),
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    max_versions: int = typer.Option(5, help="Number of recent versions to use"),
):
    
    task_scores = []
    
    # Print Header
    header = f"{'Task':<45} | {'Metric Type':<12} | {'Run Status':<35} | {'Score':<10}"
    typer.echo("\n" + header)
    typer.echo("-" * len(header))

    for task_folder, metric_config in TASK_METRIC_MAP.items():
        
        # 1. Locate Experiment Path
        exp_path = os.path.join(base_folder, task_folder, experiment_name)
        
        # 2. Get Versions (using Glob & Regex)
        found_versions = get_sorted_versions(exp_path)
        count = len(found_versions)
        
        # 3. Determine Warning Status
        if count == 0:
            status_msg = "MISSING (Found 0)"
            color = typer.colors.RED
        elif count < max_versions:
            status_msg = f"WARNING: Only {count}/{max_versions} runs"
            color = typer.colors.RED
        elif count > max_versions:
            status_msg = f"NOTE: Found {count}, used top {max_versions}"
            color = typer.colors.YELLOW
        else:
            status_msg = f"OK: Found {count}"
            color = typer.colors.GREEN

        # 4. Limit to requested versions
        target_versions = found_versions[:max_versions]
        
        # 5. Extract Scores
        current_values = []
        for v in target_versions:
            csv_path = os.path.join(exp_path, f"version_{v}", "metrics.csv")
            val = extract_raw_score(csv_path, metric_config)
            
            if val is not None:
                # --- SPECIAL LOGIC FOR AGE ---
                if task_folder == "train-age":
                    val = 1.0 - val
                # -----------------------------
                
                current_values.append(val)

        # 6. Compute Average & Display
        score_str = "N/A"
        metric_type_str = "Composite" if isinstance(metric_config, list) else "Single"

        if current_values:
            avg_val = np.mean(current_values)
            task_scores.append(avg_val)
            score_str = f"{avg_val:.5f}"
        
        # Output Row
        typer.echo(f"{task_folder:<45} | {metric_type_str:<12} | ", nl=False)
        typer.secho(f"{status_msg:<35}", fg=color, nl=False)
        typer.echo(f" | {score_str}")

    typer.echo("-" * len(header))
    
    # Final Calculation
    if task_scores:
        final_agg = np.mean(task_scores)
        typer.secho(f"\nFINAL AGGREGATED SCORE: {final_agg:.5f}", fg=typer.colors.BRIGHT_GREEN, bold=True)
    else:
        typer.secho("\nNo scores calculated.", fg=typer.colors.RED)

if __name__ == "__main__":
    app()
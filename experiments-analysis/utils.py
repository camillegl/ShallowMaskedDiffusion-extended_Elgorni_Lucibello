
import os
import pandas as pd
import yaml
import json
import torch
from diffusion import MaskedDiffusion


def load_model_and_data(run_dir):
    ckpt_path = os.path.join(run_dir, "last.ckpt")
    model = MaskedDiffusion.load_from_checkpoint(ckpt_path).cpu()
    model.eval()

    dataset_path = os.path.join(run_dir, "dataset.pt")
    dataset_dict = torch.load(dataset_path, weights_only=False)
    train_data = dataset_dict["dataset"]["data"][dataset_dict["train_idx"]].cpu()
    val_data = dataset_dict["dataset"]["data"][dataset_dict["val_idx"]].cpu()
    return model, train_data, val_data

def compute_overlaps(samples, train_data):
    overlaps = torch.zeros((samples.shape[0], train_data.shape[0]))
    for i in range(samples.shape[0]):
        for j in range(train_data.shape[0]):
            overlaps[i,j] = (samples[i] * train_data[j]).sum().item() / samples.shape[1]

    top3_q = overlaps.topk(3, dim=1).values
    return overlaps, top3_q.mean(dim=0).tolist()

def read_hparams(run_dirs):
    data = []
    for run_dir in run_dirs:
        hparams_file = os.path.join(run_dir, "hparams.yaml")
        with open(hparams_file, "r") as f:
            lines = f.readlines()
        assert lines[0].startswith("config:")
        if lines and lines[0].startswith("config: !!python/object:"):
            # Remove the tag but keep the key ("config:")
            lines[0] = "config:\n"
        hparams = yaml.safe_load("".join(lines))
        hparams["config"]["run_dir"] = run_dir
        data.append(hparams["config"])
    df = pd.DataFrame(data)

    for col in ["no_pbar", "save_dataset", "test", "eps", "seed", "lr", "pbar", "alpha_val"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    
    df = df.sort_values(by=["L", "alpha", "l2reg"]).reset_index(drop=True)
    return df

def read_test_results(run_dirs, save_fixed=False):
    data = []
    for run_dir in run_dirs:
        test_results_file = os.path.join(run_dir, "test_results.json")
        if not os.path.exists(test_results_file):
            continue
        with open(test_results_file, "r") as f:
            test_results = json.load(f)
        # assert isinstance(test_results, list) and len(test_results) == 1
        assert isinstance(test_results, list)
        test_results[0]["run_dir"] = run_dir # only consider train results for dataloader_idx_0
        
        # Fix keys from old runs
        for d in test_results:
            for key in list(d.keys()):
                if key.endswith("/dataloader_idx_0"):
                    new_key = key.replace("/dataloader_idx_0", "").replace("test/", "train/")
                    d[new_key] = d.pop(key)
                elif key.endswith("/dataloader_idx_1"):
                    new_key = key.replace("/dataloader_idx_1", "").replace("test/", "val/")
                    d[new_key] = d.pop(key)

        # Add top3 overlaps if missing
        d = test_results[0]
        if "sample/top3_overlaps_with_train" not in d:
            if os.path.exists(os.path.join(run_dir, "samples.pt")):
                samples = torch.load(os.path.join(run_dir, "samples.pt"))
                model, train_data, val_data = load_model_and_data(run_dir)
                overlaps, top3_q = compute_overlaps(samples, train_data)
                d["sample/top3_overlaps_with_train"] = top3_q

        if save_fixed:
            # resave fixed test results
            with open(test_results_file, "w") as f:
                json.dump(test_results, f, indent=4)

        data.append(test_results[0])
    df = pd.DataFrame(data)
    return df


def merge_right_prefer(left, right, on, how="outer"):
    # Merge keeping right-hand values on overlapping columns; fall back to left when right is NaN
    on_cols = [on] if isinstance(on, str) else list(on)
    overlap = [c for c in left.columns.intersection(right.columns) if c not in on_cols]
    merged = left.merge(right, on=on_cols, how=how, suffixes=("_left", "_right"))
    for c in overlap:
        rc = f"{c}_right"
        lc = f"{c}_left"
        # prefer right; if right is NaN, use left
        merged[c] = merged[rc].combine_first(merged[lc])
        merged = merged.drop(columns=[lc, rc])
    return merged

def collect_experiment_data(logs_dir = "../logs"):
    # find recursively all run dirs in the logs dir
    run_dirs = []
    for root, dirs, files in os.walk(logs_dir):	
        if "events.out.tfevents" in " ".join(files):
            run_dirs.append(root)

    df_hparams = read_hparams(run_dirs)
    df_test_results = read_test_results(run_dirs)
    df_all = merge_right_prefer(df_hparams, df_test_results, on="run_dir", how="outer")
    df_all = df_all.sort_values(by=["dataset", "model", "L", "alpha", "l2reg"]).reset_index(drop=True)
    return df_all

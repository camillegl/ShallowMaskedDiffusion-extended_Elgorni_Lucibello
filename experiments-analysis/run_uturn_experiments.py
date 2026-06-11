#!/usr/bin/env python3
"""
U-Turn Sampler Analysis Script

This script runs the U-Turn sampler on trained models and generates analysis plots.
"""

import argparse
import pandas as pd
import numpy as np
import os
import re
import torch
import sys
import pathlib

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))
from diffusion import MaskedDiffusion
from utils import collect_experiment_data


def main():
    parser = argparse.ArgumentParser(description="Run U-Turn sampler analysis")
    
    # Model selection arguments
    parser.add_argument("--L", type=int, default=2000, help="System size")
    parser.add_argument("--l2reg", type=float, default=0.01, help="L2 regularization parameter (ignored with --hebbian)")
    parser.add_argument("--dataset", type=str, default="uniform", help="Dataset name")
    parser.add_argument("--model", type=str, default="linear", help="Model type")
    parser.add_argument(
        "--hebbian",
        action="store_true",
        help="If set, analyze only runs whose run_dir contains 'hebbian' and ignore --l2reg filtering.",
    )
    parser.add_argument(
        "--include-fmw",
        action="store_true",
        help="If set, only use runs whose run_dir name contains '_fmw'. Default: exclude them.",
    )
    
    # Sampling arguments
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--t0-min", type=float, default=0.0, help="Minimum t0 value")
    parser.add_argument("--t0-max", type=float, default=1.0001, help="Maximum t0 value")
    parser.add_argument("--t0-step", type=float, default=0.02, help="Step size for t0 values")
    parser.add_argument("--decoding", type=str, default="greedy", 
                        choices=["fair", "greedy", "verygreedy"], help="Decoding strategy")
    
    # Experiment type arguments
    parser.add_argument("--uturn", action="store_true",
                        help="Run U-Turn sampler experiments (default: runs by default)")
    parser.add_argument("--oneshot", action="store_true",
                        help="Run one-shot sampling experiments")
    
    # I/O arguments
    parser.add_argument("--logs-dir", type=str, default="../logs", help="Directory containing experiment logs")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results and plots")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run sampling on. 'auto' picks CUDA > MPS > CPU",
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment data
    print("Loading experiment data...")
    df_all = collect_experiment_data(logs_dir=args.logs_dir)
    
    # Filter experiments
    if args.hebbian:
        df = df_all[
            (df_all["L"] == args.L) &
            (df_all["dataset"] == args.dataset) &
            (df_all["model"] == args.model)
        ].copy()
    else:
        df = df_all[
            (df_all["L"] == args.L) &
            (df_all["l2reg"] == args.l2reg) &
            (df_all["dataset"] == args.dataset) &
            (df_all["model"] == args.model)
        ].copy()

    # Select Hebbian/non-Hebbian runs based on run_dir naming convention.
    run_dir_col = df["run_dir"].astype(str)
    if args.hebbian:
        df = df[run_dir_col.str.contains("hebbian", na=False)].copy()
    else:
        df = df[~run_dir_col.str.contains("hebbian", na=False)].copy()

    # By default, exclude runs marked with "_fmw" in their experiment folder name.
    run_dir_col = df["run_dir"].astype(str)
    if args.include_fmw:
        df = df[run_dir_col.str.contains("_fmw", na=False)].copy()
    else:
        df = df[~run_dir_col.str.contains("_fmw", na=False)].copy()
    
    if len(df) == 0:
        if args.hebbian:
            print(
                f"No Hebbian experiments found with L={args.L}, "
                f"dataset={args.dataset}, model={args.model}, include_fmw={args.include_fmw}"
            )
        else:
            print(
                f"No experiments found with L={args.L}, l2reg={args.l2reg}, "
                f"dataset={args.dataset}, model={args.model}, include_fmw={args.include_fmw}"
            )
        return

    # If multiple runs exist for the same alpha, keep the latest version_N.
    def _version_num(run_dir: str) -> int:
        match = re.search(r"(?:^|/)version_(\d+)(?:/|$)", str(run_dir))
        if match is None:
            return -1
        return int(match.group(1))

    df["_version_num"] = df["run_dir"].apply(_version_num)
    dup_alpha = df["alpha"].value_counts()
    dup_alpha = dup_alpha[dup_alpha > 1]
    if not dup_alpha.empty:
        print("Found multiple runs for some alpha values; keeping latest version_N per alpha.")
        df = (
            df.sort_values(["alpha", "_version_num", "run_dir"])
              .groupby("alpha", as_index=False)
              .tail(1)
              .sort_values("alpha")
              .reset_index(drop=True)
        )

    df = df.drop(columns=["_version_num"])
    
    assert len(df.alpha.unique()) == len(df), "Expected one experiment per alpha value"
    print(f"Found {len(df)} experiments")
    print(df[["alpha", "run_dir"]].to_string(index=False, max_colwidth=None))
    
    # Default behavior: run both if no specific experiment type is requested
    run_uturn = args.uturn or (not args.uturn and not args.oneshot)
    exp_tag = "hebbian" if args.hebbian else f"l2reg{args.l2reg}"
    
    # Run U-Turn experiments by default or if explicitly requested
    if run_uturn:
        print("Running U-Turn sampler...")
        t0_values = np.arange(args.t0_min, args.t0_max, args.t0_step)
        
        res_uturn = []
        
        for idx, row in df.iterrows():
            alpha = row["alpha"]
            run_dir = row["run_dir"]
            
            print(f"Processing alpha={alpha:.4f}...")
            
            ckpt_path = os.path.join(run_dir, "last.ckpt")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}, skipping...")
                continue
            model = MaskedDiffusion.load_from_checkpoint(ckpt_path)

            # Select device
            if args.device == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device(args.device)

            # Move model to device and set eval mode
            model = model.to(device)
            model.eval()
            dataset_path = os.path.join(run_dir, "dataset.pt")
            dataset_dict = torch.load(dataset_path, weights_only=False)
            train_data = dataset_dict["dataset"]["data"][dataset_dict["train_idx"]].to(device)
            
            for t0 in t0_values:
                T0 = round(t0 * args.L)  # num of masked indices at the start
                x0 = train_data[torch.randint(0, train_data.shape[0], (args.num_samples,))]
                with torch.no_grad():
                    samples, history = model.mask_and_sample(x0, T0=T0, decoding_strategy=args.decoding)
                for frac_masked, frac_correct, frac_errors in history:
                    res_uturn.append([args.decoding, alpha, t0, frac_masked, frac_correct, frac_errors])

        df_uturn = pd.DataFrame(res_uturn, columns=["decoding", "alpha", "t0", "frac_masked", "frac_correct", "frac_errors"])
        df_uturn.sort_values(["alpha", "t0", "frac_masked"], inplace=True)
        df_uturn.reset_index(drop=True, inplace=True)
        
        output_file = os.path.join(args.output_dir,
            f"res-exp-uturn_{args.decoding}_L{args.L}_{exp_tag}_{args.dataset}_{args.model}_n{args.num_samples}.csv")
        if os.path.exists(output_file):
            # if exists, append a number to avoid overwriting
            base, ext = os.path.splitext(output_file)
            i = 1
            while os.path.exists(f"{base}_{i}{ext}"):
                i += 1
            output_file = f"{base}_{i}{ext}"
        df_uturn.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    # Run one-shot experiments if requested
    if args.oneshot:
        print("\n" + "=" * 80)
        print("Running one-shot sampling experiments...")
        print("=" * 80)
        
        t0_values = np.arange(args.t0_min, args.t0_max, args.t0_step)
        res_oneshot = []
        
        for idx, row in df.iterrows():
            alpha = row["alpha"]
            run_dir = row["run_dir"]
            
            print(f"Processing alpha={alpha:.4f}...")
            
            ckpt_path = os.path.join(run_dir, "last.ckpt")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}, skipping...")
                continue
            model = MaskedDiffusion.load_from_checkpoint(ckpt_path)
            dataset_path = os.path.join(run_dir, "dataset.pt")
            dataset_dict = torch.load(dataset_path, weights_only=False)
            train_data = dataset_dict["dataset"]["data"][dataset_dict["train_idx"]]
            
            for t0 in t0_values:
                T0 = round(t0 * args.L)  # num of masked indices at the start
                x0 = train_data[torch.randint(0, train_data.shape[0], (args.num_samples,))]
                samples, metrics = model.mask_and_sample_oneshot(x0, T0=T0, decoding_strategy=args.decoding)
                res_oneshot.append([
                    args.decoding, alpha, t0, 
                    metrics["frac_masked"], 
                    metrics["frac_correct"], 
                    metrics["frac_errors"]
                ])
        
        df_oneshot = pd.DataFrame(
            res_oneshot, 
            columns=["decoding", "alpha", "t0", "frac_masked", "frac_correct", "frac_errors"]
        )
        df_oneshot.sort_values(["alpha", "t0"], inplace=True)
        df_oneshot.reset_index(drop=True, inplace=True)
        
        output_file = os.path.join(args.output_dir,
            f"res-exp-oneshot_{args.decoding}_L{args.L}_{exp_tag}_{args.dataset}_{args.model}_n{args.num_samples}.csv")
        if os.path.exists(output_file):
            # if exists, append a number to avoid overwriting
            base, ext = os.path.splitext(output_file)
            i = 1
            while os.path.exists(f"{base}_{i}{ext}"):
                i += 1
            output_file = f"{base}_{i}{ext}"
        df_oneshot.to_csv(output_file, index=False)
        print(f"One-shot results saved to {output_file}")


if __name__ == "__main__":
    main()

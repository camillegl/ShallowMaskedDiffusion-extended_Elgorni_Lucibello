
import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset
import json
import socket
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import UniformIsingDataset, RandomFeaturesDataset, BinarizedMNIST
from diffusion import MaskedDiffusion


def compute_overlaps(samples1, samples2):
    overlaps = torch.zeros((samples1.shape[0], samples2.shape[0]))
    for i in range(samples1.shape[0]):
        for j in range(samples2.shape[0]):
            overlaps[i, j] = (samples1[i] * samples2[j]).sum().item() / samples1.shape[1]

    top3_q = overlaps.topk(3, dim=1).values
    return overlaps, top3_q.mean(dim=0).tolist()



def main(args):
    if args.seed >= 0:
        pl.seed_everything(args.seed)

    num_train_samples = int(round(args.alpha * args.L))
    num_val_samples = int(round(args.alpha_val * args.L))  # not used in current code
    if args.dataset == "uniform":
        Dataset = UniformIsingDataset
        num_val_samples = 0
    elif args.dataset.startswith("rf"):
        expansion_factor = float(args.dataset[2:])
        Dataset = lambda L, num_samples: RandomFeaturesDataset(n_visible=L, 
                                                               n_hidden=int(round(L / expansion_factor)),
                                                               num_samples=num_samples, 
                                                               act=torch.sign)
    elif args.dataset == "binarized_mnist":
        assert args.L == 784, "For binarized_mnist, L must be 784."
        Dataset = lambda L, num_samples: BinarizedMNIST(num_samples=num_samples, train=True)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    dataset = Dataset(L=args.L, num_samples=(num_train_samples + num_val_samples))
    train_idx = list(range(num_train_samples))
    val_idx = list(range(num_train_samples, num_train_samples + num_val_samples))
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    if args.hebbian and args.model != "linear":
        raise ValueError("Hebbian mode is only supported for model='linear'.")

    hebbian_train_data = None
    if args.hebbian:
        hebbian_train_data = torch.cat([batch for batch in train_dataloader], dim=0)

    print("Args:")
    print(json.dumps(vars(args), indent=4))
    print(f"Dataset size: train {len(train_dataset)} val {len(val_dataset)}")
    
    model = MaskedDiffusion(args)

    if args.hebbian:
        if hebbian_train_data is None:
            raise RuntimeError("Hebbian training data could not be built from the training dataloader.")
        model.backbone.set_hebbian_weights(hebbian_train_data, scale=1.0, zero_diagonal=False)

    if args.exp_dir is None:
        if "mnist" in args.dataset:
            M = len(train_dataset)
            experiment_name = f"{args.model}_{args.dataset}_L{args.L}_M{M}"
        else:
            experiment_name = f"{args.model}_{args.dataset}_L{args.L}_alpha{args.alpha}"
        if args.hebbian:
            experiment_name += "_hebbian"
        else:
            experiment_name += f"_l2reg{args.l2reg}"
        if args.bias:
            experiment_name += "_bias"
        if args.freeze_mask_weights:
            experiment_name += "_fmw"
    else:
        experiment_name = args.exp_dir
    tb_logger = TensorBoardLogger(save_dir="logs", name=experiment_name, sub_dir=socket.gethostname())
    run_dir = tb_logger.log_dir
    ckpt_callback = ModelCheckpoint(save_last=True, every_n_epochs=1, dirpath=run_dir)
    trainer = pl.Trainer(max_epochs = args.epochs,
                         callbacks = [ckpt_callback],
                         log_every_n_steps = 10,
                         logger = tb_logger, 
                         accelerator = "auto",
                         
                         enable_progress_bar = args.pbar)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if args.save_dataset:
        torch.save({"name": args.dataset,
                    "dataset": dataset.to_dict(), 
                    "train_idx": train_idx, 
                    "val_idx": val_idx}
                , f"{run_dir}/dataset.pt")

    print("Training...")
    print(f"Run directory: {run_dir}")
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    finally:
        # Always attempt to persist a last checkpoint, even on interruption/error.
        try:
            trainer.save_checkpoint(os.path.join(run_dir, "last.ckpt"))
            print(f"Saved checkpoint: {os.path.join(run_dir, 'last.ckpt')}")
        except Exception as ckpt_err:
            print(f"WARNING: failed to save last.ckpt in {run_dir}: {ckpt_err}")
    
    if args.test:
        print("Testing...")
        test_results = trainer.test(model, [train_dataloader, val_dataloader])
        # fix metric keys to indicate train/val instead of dataloader_idx
        for d in test_results:
            for k in list(d.keys()):
                if "dataloader_idx_0" in k: # always assume this is the train_dataloader
                    new_k = k.replace("test/", "train/").replace("/dataloader_idx_0", "")
                    d[new_k] = d.pop(k)
                elif "dataloader_idx_1" in k: # always assume this is the val_dataloader
                    new_k = k.replace("test/", "val/").replace("/dataloader_idx_1", "")
                    d[new_k] = d.pop(k)
        samples = model.sample(200)
        overlaps, top3_q_mean = compute_overlaps(samples, dataset.data[train_idx])
        test_results[0]["sample/top3_overlaps_with_train"] = top3_q_mean
        print("Top 3 overlaps of samples with training data:", top3_q_mean)
        torch.save(samples.cpu(), f"{run_dir}/samples.pt")
        with open(f"{run_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    def str2bool(v):
        assert v.lower() in ('true', 'false')
        return v.lower() == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="uniform", help="Dataset to use. Choices: \
                                                           'uniform', \
                                                           'rf<expansion_factor>' (random feature dataset) e.g. 'rf3.5', \
                                                           'binarized_mnist'")
    parser.add_argument("--model", type=str, default="linear", help="Model type. Choices: \
                                                            'linear', \
                                                            'rfs<expansion>-<activation>' (random feature score) e.g. 'rfs10-tanh', \
                                                            'tensor<nbins>' (linear on time slices) e.g. 'tensor4'.")
    parser.add_argument("--L", type=int, default=128, help="Dimensionality of the data.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Ratio of number of training samples to L.")
    parser.add_argument("--alpha-val", type=float, default=0.1, help="Ratio of number of validation samples to L.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2reg", type=float, default=0.0, help="L2 regularization strength.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Small constant to avoid numerical issues.")
    parser.add_argument("--seed", type=int, default=-1, help="If negative, do not set seed.")
    parser.add_argument("--test", type=str2bool, default="false", help="If set, run `trainer.test` after training.")
    parser.add_argument("--save-dataset", type=str2bool, default="true", help="If true, save the dataset in the log directory.")
    parser.add_argument("--pbar", type=str2bool, default="true", help="If true, show progress bar.")
    parser.add_argument("--exp-dir", type=str, default=None, help="Directory within the `logs` folder to save logs and checkpoints. If None (default), use auto-generated directory.")
    parser.add_argument("--hebbian", type=str2bool, default="false", help="If true, initialize the linear W matrix with the Hebbian rule from the training set and freeze it.")
    parser.add_argument("--freeze-mask-weights", type=str2bool, default="false", help="If true, freeze the weights connected to the masked variables to zero.")
    parser.add_argument("--bias", type=str2bool, default="false", help="If true, include bias terms in the model.")
    args = parser.parse_args()
    main(args)


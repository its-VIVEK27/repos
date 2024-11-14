from pprint import pprint
import argparse
import gc
from os.path import join
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from cifake_dataset import CIFAKEDataset

import model
from lib.util import load_config
import random
import numpy as np


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/ablation_baseline.cfg",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # Load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # Preliminary setup
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # Load data
    if cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        test_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            resolution=cfg["test"]["resolution"],
        )

    # Load the dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize model
    net = model.BNext4DFR.load_from_checkpoint(
        join(cfg["test"]["weights_path"], f"{cfg['dataset']['name']}_{cfg['model']['backbone'][-1]}{'_unfrozen' if not cfg['model']['freeze_backbone'] else ''}.ckpt")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # Start testing
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "DFAD_CVPRW24"
    run_label = args.cfg.split("/")[-1].split(".")[0]
    run = cfg["dataset"]["name"] + f"_test_{date}_{run_label}"
    logger = WandbLogger(project=project, name=run, id=run, log_model=False)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in str(device) else "cpu",
        devices=1,
        precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
        limit_test_batches=cfg["test"]["limit_test_batches"],
        logger=logger,
    )
    trainer.test(model=net, dataloaders=test_loader)

"""
End-to-end training entry point.

Steps:
  1. Seed RNGs (numpy / torch / cuda) so a training run is reproducible.
  2. Discover training/validation tiles and compute per-band normalisation
     statistics on the training split only — the same stats are reused for
     validation and at inference time (stored alongside the model in MLflow).
  3. Build Albumentations transforms (resize → flip → normalize → ToTensorV2).
  4. Hand the dataloaders to `pl.Trainer.fit`. Lightning drives the optimiser,
     scheduler, early-stopping and checkpointing.

Run from the project root with `python -m src.train` (or
`uv run python -m src.train`) so the `src.*` package imports resolve.
"""

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torch import Generator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.loading import load_data
from src.data.normalization import compute_global_normalization
from src.data.transforms import build_transform
from src.training.lightning import get_lightning_module
from src.data.dataset import SegmentationDataset
from src.utils import mlf_logger


# ==========================================================
# 1️⃣ CONFIG
# ==========================================================

CONFIG = {
    "train_regions": [
        "AT332",
        "BE100",
        # "BE251",
        "BG322",
        # "DEA54",
        "FRJ27",
        # "LU000",
    ],
    "train_years": ["2018", "2021"],
    "test_regions": ["BE100", "BG322"],
    "test_year": "2021",
    "batch_size": 16,
    "test_batch_size": 8,
    "epochs": 10,
    "lr": 1e-3,
    "n_bands": 14,
    "resize": 512,
    # "num_workers": os.cpu_count(),
    "seed": 42,
}


# ==========================================================
# 2️⃣ UTILS
# ==========================================================


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)


def build_region_year_list(regions, years):
    return [f"{r}_{y}" for r in regions for y in years]


# ==========================================================
# 3️⃣ MAIN PIPELINE
# ==========================================================


# Reproducibilité
set_seed(CONFIG["seed"])

# -------- Dataset IDs --------
train_ids = build_region_year_list(
    CONFIG["train_regions"],
    CONFIG["train_years"],
)

test_ids = build_region_year_list(
    CONFIG["test_regions"],
    [CONFIG["test_year"]],
)

# -------- Normalisation --------
print("📊 Computing normalization...")
mean, std = compute_global_normalization(train_ids, CONFIG["n_bands"])

# -------- Chargement --------
print("📂 Loading data...")
train_patches, train_labels = load_data(train_ids)
test_patches, test_labels = load_data(test_ids)

# -------- Transforms --------
train_transform = build_transform(mean, std, augment=True, resize=CONFIG["resize"])

test_transform = build_transform(mean, std, augment=False, resize=CONFIG["resize"])

# -------- Dataset --------
full_dataset = SegmentationDataset(
    patchs=train_patches,
    labels=train_labels,
    n_bands=CONFIG["n_bands"],
    from_s3=False,
    transform=train_transform,
)

test_dataset = SegmentationDataset(
    patchs=test_patches,
    labels=test_labels,
    n_bands=CONFIG["n_bands"],
    from_s3=False,
    transform=test_transform,
)

# -------- Split train / val --------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=Generator().manual_seed(CONFIG["seed"]),
)

# -------- DataLoaders --------
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["test_batch_size"],
    pin_memory=True,
)

# -------- Model --------
model = get_lightning_module(
    n_bands=CONFIG["n_bands"],
    lr=CONFIG["lr"],
    freeze_encoder=True,
)

# -------- Trainer --------
callbacks = [
    EarlyStopping(monitor="validation_loss", patience=5, mode="min"),
    ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    ),
]

mlf_logger.log_hyperparams({
    "lr": CONFIG["lr"],
    "batch_size": CONFIG["batch_size"],
    "epochs": CONFIG["epochs"],
    "freeze_encoder": True,
    "n_bands": CONFIG["n_bands"],
})

trainer = pl.Trainer(
    max_epochs=CONFIG["epochs"],
    accelerator="auto",
    devices="auto",
    callbacks=callbacks,
    logger=mlf_logger,
    log_every_n_steps=10,
)

# -------- Training --------
print("🔥 Training...")
trainer.fit(model, train_loader, val_loader)

print("🧪 Testing...")
trainer.test(model, test_loader)

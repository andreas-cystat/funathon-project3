"""
Factory that wires together the SegFormer backbone, the cross-entropy loss
(with `ignore_index=255` for no-data pixels) and the AdamW + OneCycleLR
schedule used in `src/train.py`. Kept as a one-call entry point so the
training script and Lightning module construction stay decoupled.
"""

from torch.nn import CrossEntropyLoss
import torch

from src.models.model import SegformerB5
from src.models.module import SegmentationModule


def get_lightning_module(
    n_bands: int,
    freeze_encoder: bool,
    lr: float,
    weights=None,
    label_smoothing: float = 0.0,
):
    """
    Create a Segmentation LightningModule with:
        - AdamW optimizer
        - ReduceLROnPlateau scheduler
    """

    model = SegformerB5(
        n_bands=n_bands,
        freeze_encoder=freeze_encoder,
    )

    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32)
    loss = CrossEntropyLoss(weight=weights)

    return SegmentationModule(
        model=model,
        loss=loss,
        optimizer=torch.optim.AdamW,
        optimizer_params={"lr": lr},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={
            "mode": "min",
            "patience": 2,
            # `monitor` matches the name logged by SegmentationModule.validation_step
            # (`self.log("validation_loss", ...)`).
            "monitor": "validation_loss",
        },
        scheduler_interval="epoch",
    )

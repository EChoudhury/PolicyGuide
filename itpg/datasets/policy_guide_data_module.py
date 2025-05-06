import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

import itpg
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, get_worker_info
import torchvision

from itpg.datasets.policy_guide_dataset import PolicyGuideDataset
from itpg.datasets.chunked_policy_guide_dataset import ChunkedPolicyGuideDataset
from itpg.datasets.utils.episode_utils import load_dataset_statistics

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})

class PolicyGuideDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        training_repo_root: Optional[Path] = None,
        root_data_dir: str = "datasets/task_D_D",
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            assert training_repo_root is not None, "If root_data_path isn't absolute, please provide training_repo_root"
            root_data_path = training_repo_root / root_data_path
        self.abs_datasets_dir = root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        # Initialize transforms
        self.transforms = transforms        

    def setup(self, stage=None):
        print("Loading Data...╰( ͡° ͜ʖ ͡°)つ──☆*:・ﾟ")
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        # Load and instantiate transforms
        train_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]]
            for cam in self.transforms.train
        }
        val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]]
            for cam in self.transforms.val
        }

        # Compose transforms
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in val_transforms.items()}

        # Instantiate datasets with transforms
        self.train_dataset = hydra.utils.instantiate(
            self.datasets_cfg, dataset_dir=self.training_dir, abs_datasets_dir=self.abs_datasets_dir, transforms=self.train_transforms
        )
        self.val_dataset = hydra.utils.instantiate(
            self.datasets_cfg, dataset_dir=self.val_dir, abs_datasets_dir=self.abs_datasets_dir, transforms=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.datasets_cfg.batch_size,
                num_workers=self.datasets_cfg.num_workers,
                pin_memory=False,
                shuffle=False,
            )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.datasets_cfg.batch_size,
                num_workers=self.datasets_cfg.num_workers,
                pin_memory=False,
                shuffle=False,
            )
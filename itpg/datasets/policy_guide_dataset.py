from typing import Any, Dict, Callable
import os
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import Dataset

from itpg.datasets.utils.robot_replay_buffer import RobotReplayBuffer
from itpg.datasets.utils.sampler import SequenceSampler


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value).float()
    return result


class PolicyGuideDataset(Dataset):
    """
    Dataset Class for Policy Guide

    Args:
        datasets_dir: Path of folder containing episode files.
        n_obs_steps: Number of observations used.
        horizon: The predicted action horizon.
        n_action_steps: Number of executed action steps.
        pad_before: Apply padding before episode.
        pad_after: Apply padding after episode.
        
    """
    def __init__(
            self,
            obs_space: DictConfig,
            proprio_state: DictConfig,
            dataset_dir: str,
            abs_datasets_dir: str,
            lang_folder: str,
            transforms: Dict = {},
            n_obs_steps: int = 2,
            horizon: int = 16,
            n_action_steps: int = 8,
            pad_before: int = 0,
            pad_after: int = 0,
            *args: Any,
            **kwargs: Any,
        ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        
        self.datasets_dir = dataset_dir
        self.abs_datasets_dir = dataset_dir
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.lang_folder = lang_folder 

        self.keys = [
            'language', 
            'actions', 
            'rgb_gripper', 
            'rgb_static', 
            'robot_obs', 
            'episode_step'
        ]

        self.replay_buffer = RobotReplayBuffer.copy_from_path(self.datasets_dir, self.keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
        )

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image = np.moveaxis(sample['rgb_static'],-1,1)/255
        data = {
            # Image: B, 200, 200, 3
            #TODO: Rearrange the image axis to be (B, 3, 200, 200)
            'observation.image_static': sample['rgb_static'][:self.n_obs_steps,:], 
            # State: B, 8
            'observation.state': sample['robot_obs'][:self.n_obs_steps,:],
            # Actions: B, 8
            'action': sample['actions'],
            # Annotation index
            'language': sample['language'][:1],
        }

        return data

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
         # Apply transforms if available
        if self.transforms:
            if "observation.image_static" in self.transforms:
                torch_data["observation.image_static"] = self.transforms["observation.image_static"](torch_data["observation.image_static"])
            if "observation.state" in self.transforms:
                torch_data["observation.state"] = self.transforms["observation.state"](torch_data["observation.state"])
            if "action" in self.transforms:
                torch_data["action"] = self.transforms["action"](torch_data["action"])
        return torch_data
        
if __name__ == "__main__":
    dataset = PolicyGuideDataset(
        datasets_dir="/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training"
    )
    i = 20
    data = dataset[i]
    print(len(dataset))
    print("obs: ", data['observation.image_static'])
    print("action: ", data['action'])
    print("observation.state: ", data['observation.state'])
    print("obs: ", data['observation.image_static'].shape)
    print("action: ", data['action'].shape)
    print("observation.state: ", data['observation.state'].shape)
    print("done")

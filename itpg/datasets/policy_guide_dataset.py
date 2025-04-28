from typing import Any, Dict, Callable
import os
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import Dataset
from itpg.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

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
        data = {
            # Image: B, 200, 200, 3
            'observation.image_static': sample['rgb_obs']["rgb_static"][:self.n_obs_steps,:], 
            # Image: B, 84, 84, 3
            # 'observation.image_wrist': sample['rgb_obs']["rgb_gripper"][:self.n_obs_steps,:], 
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
        data = self._get_sequences(sample)
        torch_data = self._sample_to_data(data)
        return torch_data
        
    def _get_sequences(self, episode) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        # seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        # info = get_state_info_dict(episode)
        # seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_lang = {"language": episode["language"][:1]}
        # info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_acts, **seq_lang}  # type:ignore
        # seq_dict["idx"] = idx  # type:ignore

        return seq_dict

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

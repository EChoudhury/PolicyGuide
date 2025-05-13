from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import zarr
import torch
import random

from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from itpg.datasets.utils.episode_utils import process_actions, process_rgb, process_state
from itpg.datasets.utils.robot_replay_buffer import RobotReplayBuffer
from itpg.datasets.utils.sampler import SequenceSampler

class ChunkedPolicyGuideDataset(IterableDataset):
    def __init__(self, 
                obs_space: DictConfig,
                proprio_state: DictConfig,
                dataset_dir: str,
                num_workers: int,
                abs_datasets_dir: str,
                lang_folder: str,
                transforms: Dict = {},
                n_obs_steps: int = 2,
                horizon: int = 16,
                n_action_steps: int = 8,
                pad_before: int = 0,
                pad_after: int = 0,
                chunk_size: int = 256,
                episode_length: int = 60,
                *args: Any,
                **kwargs: Any,):
        
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.relative_actions = "rel_actions" in obs_space["actions"]

        self.datasets_dir = dataset_dir
        self.abs_datasets_dir = abs_datasets_dir
        self.num_workers = num_workers
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.lang_folder = lang_folder
        self.chunk_size = chunk_size
        self.episode_length = episode_length

        self.keys = [
            'language',
            'actions',
            'rgb_gripper',
            'rgb_static',
            'robot_obs',
            'episode_step'
        ]
        self.chunk_size = chunk_size

        # Load the replay buffer metadata
        self.replay_buffer = RobotReplayBuffer.create_from_path(self.datasets_dir, mode='r')
        self.total_size = len(self.replay_buffer.episode_ends)
        # print(f'Total Size: {self.total_size}')

    def __len__(self):
        # print(f'__len__: {self.total_size}')
        return self.total_size * self.episode_length #len(self.sampler) #

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Assign a section of the dataset to this worker
        # per_worker = int(np.ceil(self.total_size / num_workers))
        # start_idx = worker_id * per_worker
        # end_idx = min(start_idx + per_worker, self.total_size)
        per_worker = self.total_size // num_workers
        remainder = self.total_size % num_workers

        start_idx = worker_id * per_worker + min(worker_id, remainder)
        end_idx = start_idx + per_worker + (1 if worker_id < remainder else 0)

        for chunk_start in range(start_idx, end_idx, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, end_idx)

            # print(f"worker id: {worker_info.id}")
            print(f"chunk_start: {start_idx}, chunk_end: {end_idx}, sub_start_chunk: {chunk_start}, sub_end_chunk: {chunk_end}")

            episode_mask = np.zeros(self.total_size, dtype=bool)
            episode_mask[chunk_start:chunk_end] = True

            sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=episode_mask,
            )

            for idx in range(len(sampler)):
                sample = sampler.sample_sequence(idx)
                data = self._get_sequences(sample)
                yield self._sample_to_data(data)


    def _sample_to_data(self, sample):
        """Convert a sample to the desired data format."""
        data = {
            'observation.image_static': sample['rgb_obs']["rgb_static"][:self.n_obs_steps, :],
            'observation.image_wrist': sample['rgb_obs']["rgb_gripper"][:self.n_obs_steps,:], 
            'observation.state': sample['robot_obs'][:self.n_obs_steps, :],
            'action': sample['actions'],
            'language': sample['language'][:1],
        }
        return data
    
    def _get_sequences(self, episode) -> Dict:
        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        seq_lang = {"language": episode["language"][:1]}
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_acts, **seq_lang}
        return seq_dict
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig
from itpg.datasets.utils.robot_replay_buffer import RobotReplayBuffer
from itpg.datasets.utils.sampler import SequenceSampler


class PolicyGuideDataset(Dataset):
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
        chunk_size: int = 10,  # Number of episodes to load at a time
        *args: Any,
        **kwargs: Any,
    ):
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.relative_actions = "rel_actions" in obs_space["actions"]

        self.datasets_dir = dataset_dir
        self.abs_datasets_dir = abs_datasets_dir
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.lang_folder = lang_folder
        self.chunk_size = chunk_size

        self.keys = [
            'language',
            'actions',
            'rgb_gripper',
            'rgb_static',
            'robot_obs',
            'episode_step'
        ]

        # Load the replay buffer metadata
        self.replay_buffer = RobotReplayBuffer.copy_from_path(self.datasets_dir, self.keys)
        self.n_episodes = len(self.replay_buffer.episode_ends)

        # Initialize chunk management
        self.current_chunk_start = 0
        self.current_chunk_end = min(self.chunk_size, self.n_episodes)
        self._load_current_chunk()

    def _load_current_chunk(self):
        """Load the current chunk of episodes into the sampler."""
        episode_mask = np.zeros(self.n_episodes, dtype=bool)
        episode_mask[self.current_chunk_start:self.current_chunk_end] = True

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
        )

    def _advance_chunk(self):
        """Advance to the next chunk of episodes."""
        self.current_chunk_start = self.current_chunk_end
        self.current_chunk_end = min(self.current_chunk_start + self.chunk_size, self.n_episodes)
        if self.current_chunk_start < self.n_episodes:
            self._load_current_chunk()

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        if idx >= len(self.sampler):
            self._advance_chunk()
            idx = idx % len(self.sampler)  # Adjust index for the new chunk
        sample = self.sampler.sample_sequence(idx)
        return self._sample_to_data(sample)

    def _sample_to_data(self, sample):
        """Convert a sample to the desired data format."""
        data = {
            'observation.image_static': sample['rgb_obs']["rgb_static"][:self.n_obs_steps, :],
            'observation.state': sample['robot_obs'][:self.n_obs_steps, :],
            'action': sample['actions'],
            'language': sample['language'][:1],
        }
        return data
# from typing import Dict, Callable
# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset

# from itpg.datasets.utils.robot_replay_buffer import RobotReplayBuffer
# from itpg.datasets.utils.sampler import SequenceSampler


# def dict_apply(
#         x: Dict[str, torch.Tensor], 
#         func: Callable[[torch.Tensor], torch.Tensor]
#         ) -> Dict[str, torch.Tensor]:
#     result = dict()
#     for key, value in x.items():
#         if isinstance(value, dict):
#             result[key] = dict_apply(value, func)
#         else:
#             result[key] = func(value)
#     return result


# class PolicyGuideDataset(Dataset):
#     """
#     Dataset Class for Policy Guide

#     Args:
#         datasets_dir: Path of folder containing episode files.
#         n_obs_steps: Number of observations used.
#         horizon: The predicted action horizon.
#         n_action_steps: Number of executed action steps.
#         pad_before: Apply padding before episode.
#         pad_after: Apply padding after episode.
        
#     """
#     def __init__(
#             self,
#             datasets_dir: str,
#             n_obs_steps: int = 2,
#             horizon: int = 16,
#             n_action_steps: int = 8,
#             pad_before: int = 0,
#             pad_after: int = 0,
#         ):
        
#         self.datasets_dir = datasets_dir
#         self.n_obs_steps = n_obs_steps
#         self.horizon = horizon
#         self.n_action_steps = n_action_steps
#         self.pad_before = pad_before
#         self.pad_after = pad_after

#         self.keys = [
#             'language', 
#             'actions', 
#             'rgb_gripper', 
#             'rgb_static', 
#             'robot_obs', 
#             'episode_step'
#         ]

#         self.replay_buffer = RobotReplayBuffer.copy_from_path(self.datasets_dir, self.keys)

#         self.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer, 
#             sequence_length=horizon,
#             pad_before=pad_before, 
#             pad_after=pad_after,
#         )

#     def __len__(self):
#         return len(self.sampler)

#     def _sample_to_data(self, sample):
#         image = np.moveaxis(sample['rgb_static'],-1,1)/255
#         data = {
#             'observation': {
#                 # Image: B, 3, 200, 200
#                 'image_static': image, 
#                  # State: B, 8
#                 'state': sample['robot_obs'].astype(np.float32),
#             },
#             'language': sample['language'].astype(np.int16),
#              # Actions: B, 8
#             'action': sample['actions'].astype(np.float32)
#         }
#         return data

#     def __getitem__(self, idx):
#         sample = self.sampler.sample_sequence(idx)
#         data = self._sample_to_data(sample)
#         torch_data = dict_apply(data, torch.from_numpy)
#         return torch_data
        
# if __name__ == "__main__":
#     dataset = PolicyGuideDataset(
#         datasets_dir="/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training"
#     )
#     i = 20
#     data = dataset[i]
#     print("obs: ", data['observation']['image_static'])
#     print("action: ", data['action'])
#     print("language: ", data['language'])
#     print("done")

import torch
import zarr
import numpy as np
from typing import Any, Dict, Callable

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value).half()
    return result

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# dataset
class PolicyGuideDataset(torch.utils.data.Dataset):
    def __init__(self,
                 *args: Any,
                 dataset_dir: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 **kwargs: Any,
                ):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_dir, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['rgb_static'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)

        # (N, D)
        action_data = dataset_root['data']['actions'][:]
        robot_obs_data = dataset_root['data']['robot_obs'][:]
        episode_ends = dataset_root['meta']['episode_ends'][:]

        train_data = dict()
        train_data['rgb_static'] = train_image_data
        train_data['actions'] = action_data
        train_data['robot_obs'] = robot_obs_data
        train_data['episode_step'] = episode_ends

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        self.indices = indices
        self.train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get data using these indices
        sample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        data = {
            # Image: B, 3, 200, 200
            'observation.image_static': sample['rgb_static'][:self.obs_horizon,:], 
            # State: B, 8
            'observation.state': sample['robot_obs'][:self.obs_horizon,:],
            # Actions: B, 8
            'action': sample['actions']
        }

        # convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    dataset = PolicyGuideDataset(
        dataset_dir = "/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training",
        pred_horizon = 16,
        obs_horizon = 6,
        action_horizon = 8
    )
    i = 20
    print(len(dataset))
    data = dataset[i]
    print(len(data))
    print("rgb_image: ", data['observation']['images'].shape)
    print("state: ", data['observation']['state'].shape)
    print("action: ", data['action'].shape)
    print("obs: ", type(data['observation']['images']))
    print("state: ", type(data['observation']['state']))
    print("action: ", type(data['action']))

    # print("language: ", data['language'])
    print("done")
#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
from typing import Dict


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def convert_batch(self, batch: Dict[str, Dict], train=True, infer=False) -> Dict[str, torch.Tensor]:
        """
        Convert the batch dictionary into the desired format.

        Args:
            batch (dict): Input batch dictionary.

        Returns:
            dict: Converted batch dictionary.
        """
        if infer:
            B = 1
            torch.concat(batch)
        else:
            B = len(batch['idx'])  # Batch size
            
        n_obs_steps = batch['robot_obs'].shape[1]  # Number of observation steps
        state_dim = batch['robot_obs'].shape[2]  # State dimension

        # Assuming same image size for all cameras
        C, H, W = batch['rgb_obs']['rgb_static'].shape[2:]  # Channels, height, and width of images
        
        converted_batch = {
            "observation.state": batch['robot_obs'].view(B, n_obs_steps, state_dim),
            "observation.image_static": batch['rgb_obs']['rgb_static'].view(B, n_obs_steps, C, H, W),
        }

        converted_batch["observation.image_static"] = converted_batch["observation.image_static"][:,:self.config.n_obs_steps,...]
        converted_batch["observation.state"] = converted_batch["observation.state"][:,:self.config.n_obs_steps,:]

        if train:
            action_dim = batch['actions'].shape[-1]
            converted_batch["action"] = batch['actions'].view(B, n_obs_steps, action_dim)  # Assuming actions have shape (B, n_obs_steps, action_dim)
            converted_batch["action"] = converted_batch["action"][:, :self.config.horizon, :]

        # if infer:
        #     print(converted_batch["observation.image_static"].shape, converted_batch["observation.state"].shape, converted_batch["action"].shape)
        
        return converted_batch

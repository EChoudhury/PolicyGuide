import logging
import os
from pathlib import Path
import pickle
import cv2
from typing import Any, Dict, Optional, Tuple

from itpg.policy.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D

from itpg.policy.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from itpg.policy.models.diffusion_policy.configuration_diffusion import DiffusionConfig
from itpg.utils.utils import get_aff_model, get_abspath

from itpg.affordance.dataset_creation.core.utils import instantiate_test_env

logger = logging.getLogger(__name__)

class ITPG(pl.LightningModule, CalvinBaseModel):
    """
    The lightning module used for training.

    Args:
        optimizer: DictConfig for optimizer.
    """

    def __init__(
        self,
        optimizer: DictConfig,
        affordance_checkpoint: dict,
        replan_freq: int,
        stats_path: str,
    ):
        super(ITPG, self).__init__()

        # affordance policy

        # diffusion policy network
        # load stats dataset for normalization
        self.stats = self._get_stats(stats_path)
        # self.diffusion_policy = hydra.utils.instantiate(diffusion_policy)
        self.config = DiffusionConfig()
        self.diffusion_policy = DiffusionPolicy(self.config, self.stats)

        # affordance policy network
        print(affordance_checkpoint)
        self.camera = self._get_camera("/home/choudhue/PolicyGuide/hydra_outputs/datacollection/2025-03-24_16-05-27")
        self.affordance_policy, _ = get_aff_model(**affordance_checkpoint)
        self.affordance_policy = self.affordance_policy.cuda()

        self.modality_scope = "vis"
        self.optimizer_config = optimizer

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

        for param in self.diffusion_policy.parameters():
            param.requires_grad = True

        for param in self.diffusion_policy.parameters():
            assert param.requires_grad, "Parameter does not require gradients"
        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        self.lang_embeddings = None


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer


    def _get_stats(self, stats_path: str):
        print(f"Retrieving stats data from {stats_path}...")
        with open(stats_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        print("############## Successfully loaded stats data ##############")
        print(loaded_data)
        print("############################################################")
        return loaded_data
    

    def _get_camera(self, path):
        play_data_hydra_cfg = path + "/.hydra"
        play_data_hydra_cfg = get_abspath(play_data_hydra_cfg)
        play_data_cfg = OmegaConf.load(play_data_hydra_cfg + "/config.yaml")
        static_cam, _ = instantiate_test_env(play_data_cfg, "simulation")
        return static_cam


    def _convert_batch(self, batch: Dict[str, Dict], train=True, infer=False) -> Dict[str, torch.Tensor]:
        """
        Convert the batch dictionary into the desired format.

        Args:
            batch (dict): Input batch dictionary.

        Returns:
            dict: Converted batch dictionary.
        """
        if infer:
            B = 1
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

        # print(converted_batch["observation.image_static"].shape, converted_batch["observation.state"].shape)
        
        return converted_batch
    

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        # convert observations
        converted_obs = self._convert_batch(batch["vis"])
        
        # Run through diffusion policy
        loss = self.diffusion_policy.forward(converted_obs)
        
        self.log("train/total_loss", loss, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing losses and the sampled plans of plan recognition and plan proposal networks.
        """
        # convert observations
        converted_obs = self._convert_batch(batch["vis"])

        # Run inference on diffusion policy
        loss = self.diffusion_policy.forward(converted_obs)

        self.log("valid/valid_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        for i, step in enumerate(validation_step_outputs):
            self.log(f"val_loss/step_{i}", step)


    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        
        converted_obs = self._convert_batch(obs, train=False, infer=True)

        # replan every replan_freq steps (default 30 i.e every second)
        # padded_guide = None
        # if self.rollout_step_counter % self.replan_freq == 0:
            # Not using language goal for now
            # convert observations
            # Use affordance model to get the guide
        frame = converted_obs["observation.image_static"][:,-1,...].detach().cpu().numpy()
        frame = frame.squeeze()
        frame = (frame * 255.0).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.resize(frame, ([224, 224]))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        affordance_obs = {"img": frame, "lang_goal": goal}
        afford_pred = self.affordance_policy.predict(affordance_obs)
        guide = self.camera.deproject_single_depth(afford_pred["pixel"], afford_pred["depth"])
        padded_guide = np.concat((guide, np.array([0, 0, 1.5707963, 1])))
        padded_guide = torch.tensor(padded_guide).cuda()
        padded_guide = torch.unsqueeze(padded_guide, 0)

        # Run inference on diffusion policy
        action = self.diffusion_policy.run_inference(converted_obs, guide=padded_guide)

        self.rollout_step_counter += 1
        return action, padded_guide

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}


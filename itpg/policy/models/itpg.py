import logging
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple

from itpg.policy.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D

from itpg.policy.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from itpg.policy.models.diffusion_policy.configuration_diffusion import DiffusionConfig

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
        replan_freq: int = 30,
    ):
        super(ITPG, self).__init__()

        # affordance policy

        # diffusion policy network
        # load stats dataset for normalization
        stats_path = Path("/home/choudhue/PolicyGuide/dataset/calvin_debug_dataset/stats") / "calvin_debug_dataset_stats.pkl"
        self.stats = self._get_stats(stats_path)
        # self.diffusion_policy = hydra.utils.instantiate(diffusion_policy)
        config = DiffusionConfig()
        self.diffusion_policy = DiffusionPolicy(config, self.stats)

        self.modality_scope = "vis"
        self.optimizer_config = optimizer

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

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
    

    def _convert_batch(self, batch: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert the batch dictionary into the desired format.

        Args:
            batch (dict): Input batch dictionary.

        Returns:
            dict: Converted batch dictionary.
        """
        # Extract dimensions
        # print(batch.keys())
        
        B = len(batch['idx'])  # Batch size
        n_obs_steps = batch['robot_obs'].shape[1]  # Number of observation steps
        state_dim = batch['robot_obs'].shape[2]  # State dimension
        num_cameras = 1  # Number of cameras (rgb_static and rgb_wrist)

        # Assuming same image size for all cameras
        C, H, W = batch['rgb_obs']['rgb_static'].shape[2:]  # Channels, height, and width of images
        # Reshape tensors
        # converted_batch = {
        #     "observation.state": batch['robot_obs'].view(B, n_obs_steps, state_dim),
        #     "observation.images": torch.stack([
        #         batch['rgb_obs']['rgb_static'],
        #         batch['rgb_obs']['rgb_wrist']
        #     ], dim=2).view(B, n_obs_steps, num_cameras, C, H, W),
        #     "action": batch['actions'].view(B, n_obs_steps, -1)  # Assuming actions have shape (B, n_obs_steps, action_dim)
        # }
        converted_batch = {
            "observation.state": batch['robot_obs'].view(B, n_obs_steps, state_dim),
            "observation.image_static": batch['rgb_obs']['rgb_static'].view(B, n_obs_steps, C, H, W),
            "action": batch['actions'].view(B, n_obs_steps, -1)  # Assuming actions have shape (B, n_obs_steps, action_dim)
        }
        print(f"Action shape: {converted_batch['action'].shape}")

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
        total_loss = torch.tensor(0.0).to(self.device)

        for self.modality_scope, dataset_batch in batch.items():
            print(f"Modality Scope: {self.modality_scope}")

            # convert observations
            converted_obs = self._convert_batch(dataset_batch)
            
            # Run through diffusion policy
            loss = self.diffusion_policy.forward(converted_obs)
            
            total_loss += loss
            self.log(f"train/total_loss_{self.modality_scope}", loss, on_step=False, on_epoch=True)

        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)

        return total_loss


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
        output = {}
        for self.modality_scope, dataset_batch in batch.items():
            # convert observations
            converted_obs = self._convert_batch(dataset_batch)
            # Run inference on diffusion policy
            output = self.diffusion_policy.run_inference(converted_obs)
        return output

    def validation_epoch_end(self, validation_step_outputs):
        val_total_act_loss_pr = torch.tensor(0.0).to(self.device)
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        val_kl_loss = torch.tensor(0.0).to(self.device)
        val_total_mae_pr = torch.tensor(0.0).to(self.device)
        val_total_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pr = torch.tensor(0.0).to(self.device)
        val_orn_mae_pp = torch.tensor(0.0).to(self.device)
        val_orn_mae_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pp = torch.tensor(0.0).to(self.device)
        for mod in self.trainer.datamodule.modalities:
            act_loss_pp = torch.stack([x[f"val_action_loss_pp_{mod}"] for x in validation_step_outputs]).mean()
            act_loss_pr = torch.stack([x[f"val_action_loss_pr_{mod}"] for x in validation_step_outputs]).mean()
            kl_loss = torch.stack([x[f"kl_loss_{mod}"] for x in validation_step_outputs]).mean()
            mae_pp = torch.cat([x[f"mae_pp_{mod}"] for x in validation_step_outputs])
            mae_pr = torch.cat([x[f"mae_pr_{mod}"] for x in validation_step_outputs])
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            grip_sr_pp = torch.stack([x[f"gripper_sr_pp{mod}"] for x in validation_step_outputs]).mean()
            grip_sr_pr = torch.stack([x[f"gripper_sr_pr{mod}"] for x in validation_step_outputs]).mean()
            val_total_mae_pr += pr_mae_mean
            val_total_mae_pp += pp_mae_mean
            val_pos_mae_pp += pos_mae_pp
            val_pos_mae_pr += pos_mae_pr
            val_orn_mae_pp += orn_mae_pp
            val_orn_mae_pr += orn_mae_pr
            val_grip_sr_pp += grip_sr_pp
            val_grip_sr_pr += grip_sr_pr
            val_total_act_loss_pp += act_loss_pp
            val_total_act_loss_pr += act_loss_pr
            val_kl_loss += kl_loss

            self.log(f"val_act/{mod}_act_loss_pp", act_loss_pp, sync_dist=True)
            self.log(f"val_act/{mod}_act_loss_pr", act_loss_pr, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pr", grip_sr_pr, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pp", grip_sr_pp, sync_dist=True)
            self.log(f"val_kl/{mod}_kl_loss", kl_loss, sync_dist=True)
        self.log(
            "val_act/action_loss_pp", val_total_act_loss_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_act/action_loss_pr", val_total_act_loss_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_kl/kl_loss", val_kl_loss / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log(
            "val_total_mae/total_mae_pr", val_total_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_total_mae/total_mae_pp", val_total_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_pos_mae/pos_mae_pr", val_pos_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_pos_mae/pos_mae_pp", val_pos_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pr", val_orn_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pp", val_orn_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pr", val_grip_sr_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pp", val_grip_sr_pp / len(self.trainer.datamodule.modalities), sync_dist=True)

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
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
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            # Not using language goal for now
            # convert observations
            converted_obs = self._convert_batch(obs)

            # Run inference on diffusion policy
            action = self.diffusion_policy.run_inference(converted_obs)

        self.rollout_step_counter += 1
        return action

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}


import logging
import os
from pathlib import Path
import pickle
import cv2
from tqdm import tqdm 
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
        language_encoder: DictConfig,
        affordance_checkpoint: dict,
        replan_freq: int,
        root_data_dir: str,
        stats_path: None = None,
        use_affordance: bool = False,
        affordance_duration: int = 10,
        use_lang_encoding: bool = True,
    ):
        super(ITPG, self).__init__()

        # affordance toggle
        self.use_affordance = use_affordance
        self.affordance_duration = affordance_duration

        # language encoder toggle
        self.use_lang_encoding = use_lang_encoding

        # load stats dataset for normalization
        self.stats = self._get_stats(stats_path)

        # TODO: move config file to hydra
        # self.diffusion_policy = hydra.utils.instantiate(diffusion_policy)

        # load diffusion policy
        self.config = DiffusionConfig()
        self.diffusion_policy = DiffusionPolicy(self.config, self.stats)

        # load affordance policy network
        self.camera = self._get_camera(affordance_checkpoint.merged_folder)
        self.affordance_policy, _ = get_aff_model(**affordance_checkpoint.model)
        self.affordance_policy = self.affordance_policy.cuda()

        # load language encoder
        self.language_encoder = hydra.utils.instantiate(language_encoder)
        self.language_encoder = self.language_encoder.cuda()

        # load optimizer
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

        self._wandb_watch_called = False

        # load language annotations if using encodings
        self.root_data_dir = root_data_dir
        if self.use_lang_encoding:
            self.train_lang_annotations = self.load_lang_annotations(root_data_dir + "/training/lang_annotations/auto_lang_ann.npy")
            self.val_lang_annotations = self.load_lang_annotations(root_data_dir + "/validation/lang_annotations/auto_lang_ann.npy")


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if not self._wandb_watch_called:
            wandb.watch(self, log="all", log_freq=100)
            self._wandb_watch_called = True 
        return optimizer


    def _get_stats(self, stats_path: str):
        if stats_path is None:
            print(f"No statistics path included. Continuing with no normalization...")
            return None
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
            "observation.image_wrist": batch['rgb_obs']['rgb_gripper'].view(B, n_obs_steps, C, H, W),
        }

        converted_batch["observation.image_static"] = converted_batch["observation.image_static"][:,:self.config.n_obs_steps,...]
        converted_batch["observation.image_wrist"] = converted_batch["observation.image_wrist"][:,:self.config.n_obs_steps,...]
        converted_batch["observation.state"] = converted_batch["observation.state"][:,:self.config.n_obs_steps,:]

        if train:
            action_dim = batch['actions'].shape[-1]
            converted_batch["action"] = batch['actions'].view(B, n_obs_steps, action_dim)  # Assuming actions have shape (B, n_obs_steps, action_dim)
            converted_batch["action"] = converted_batch["action"][:, :self.config.horizon, :]
        
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
        # get text encodings or remove language
        if self.use_lang_encoding:
            encodings = []
            for idx in batch["language"]:
                ann = self.train_lang_annotations[int(idx[0])]
                encodings.append(self.language_encoder.encode_text(ann)[0])
            batch["observation.embedding"] = torch.stack(encodings).squeeze()
        else:
            batch.pop("language", None) 

        # Run through diffusion policy
        loss = self.diffusion_policy.forward(batch)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss


    # def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
    #     """
    #     Compute and log the validation losses and additional metrics.

    #     Args:
    #         batch (dict):
    #             - 'vis' (dict):
    #                 - 'rgb_obs' (dict):
    #                     - 'rgb_static' (Tensor): RGB camera image of static camera
    #                     - ...
    #                 - 'depth_obs' (dict):
    #                     - 'depth_static' (Tensor): Depth camera image of depth camera
    #                     - ...
    #                 - 'robot_obs' (Tensor): Proprioceptive state observation.
    #                 - 'actions' (Tensor): Ground truth actions.
    #                 - 'state_info' (dict):
    #                     - 'robot_obs' (Tensor): Unnormalized robot states.
    #                     - 'scene_obs' (Tensor): Unnormalized scene states.
    #                 - 'idx' (LongTensor): Episode indices.
    #             - 'lang' (dict):
    #                 Like 'vis' but with additional keys:
    #                     - 'language' (Tensor): Embedded Language labels.
    #                     - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
    #                         auxiliary loss.
    #         batch_idx (int): Integer displaying index of this batch.

    #     Returns:
    #         Dictionary containing losses and the sampled plans of plan recognition and plan proposal networks.
    #     """
    #      # get text encodings or remove language
    #     if self.use_lang_encoding:
    #         encodings = []
    #         for idx in batch["language"]:
    #             ann = self.val_lang_annotations[int(idx[0])]
    #             encodings.append(self.language_encoder.encode_text(ann)[0])
    #         batch["observation.embedding"] = torch.stack(encodings).squeeze()
    #     else:
    #         batch.pop("language", None) 

    #     # Run validation on diffusion policy
    #     loss = self.diffusion_policy.forward(batch)

    #     # log validation loss
    #     self.log("valid/loss", loss, on_step=False, on_epoch=True)

    #     return loss


    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.rollout_step_counter = 0

    def step(self, obs, goal=None, last_action=None):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        # convert observations
        converted_obs = self._convert_batch(obs, train=False, infer=True)

        # get text encodings
        if self.use_lang_encoding:
            converted_obs["observation.embedding"] = self.language_encoder.encode_text(goal)[0]

        # temporary hardcoded goal
        # goal = "use the switch to turn on the light bulb"
        # "pull the handle to open the drawer"
        # "press the button to turn on the led light"
        # "use the switch to turn on the light bulb"

        padded_guide = None

        # Use affordance model to get the guide
        if self.use_affordance:
            if self.rollout_step_counter % self.replan_freq == 0 and self.rollout_step_counter < self.affordance_duration:
                frame = converted_obs["observation.image_static"][:,-1,...].detach().cpu().numpy()
                frame = frame.squeeze()
                frame = (frame * 255.0).astype("uint8")
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv2.resize(frame, ([224, 224]))
                affordance_obs = {"img": frame, "lang_goal": goal}
                afford_pred = self.affordance_policy.predict(affordance_obs)
                guide = self.camera.deproject_single_depth(afford_pred["pixel"], afford_pred["depth"])
                padded_guide = np.concat((guide, last_action[3:]))
                padded_guide = torch.tensor(padded_guide).cuda()
                padded_guide = torch.unsqueeze(padded_guide, 0)

        # padded_guide = None
        # Run inference on diffusion policy
        action = self.diffusion_policy.run_inference(converted_obs, guide=padded_guide)

        self.rollout_step_counter += 1
        return action, padded_guide

    def load_lang_annotations(self, annotations_path):
        """
        This has to be called before training with language. Loads the lang annotations from the dataset.

        Args:
            annotations_path: Path to <dataset>/[training/validation]/lang_annotations/auto_lang_ann.npy
        """
        annotations = np.load(annotations_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        lang_annotations = {
            k: v for k, v in tqdm(enumerate(annotations["language"]["ann"]), desc=f"Loading annotations from {annotations_path}")
        }
        return lang_annotations


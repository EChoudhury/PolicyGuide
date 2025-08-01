import logging
import os
from pathlib import Path
import pickle
import gzip
import cv2
from tqdm import tqdm 
from typing import Any, Dict, Optional, Tuple
import time
import wandb

from itpg.policy.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D
from sentence_transformers import util

from itpg.policy.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from itpg.policy.models.diffusion_policy.configuration_diffusion import DiffusionConfig
from itpg.utils.utils import get_aff_model, get_abspath
# from itpg.policy.models.language_encoders.chatgpt_describer import ChatGPTDescriber

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
        use_affordance: bool = True,
        affordance_duration: int = 10,
        use_lang_encoding: bool = False,
        normalize_language_embeddings: bool = True,
    ):
        super(ITPG, self).__init__()

        self.cuda_device = 3

        # affordance toggle
        self.use_affordance = True #use_affordance
        self.affordance_duration = 30 #affordance_duration

        # language encoder toggle
        self.use_lang_encoding = False #use_lang_encoding
        self.normalize_language_embeddings = normalize_language_embeddings

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
        self.affordance_policy = self.affordance_policy.cuda(self.cuda_device)

        # load language encoder
        # if self.use_lang_encoding:
        self.language_encoder = hydra.utils.instantiate(language_encoder)
        self.language_encoder = self.language_encoder.cuda(self.cuda_device)

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
        self.guide = None
        self.trajectory_selection = None

        self.use_chatgpt = False
        self.chatgpt_describer = None
        # if self.use_chatgpt:
        #     self.chatgpt_describer = ChatGPTDescriber(
        #         model="gpt-4o-mini-2024-07-18",
        #         api_key=os.getenv("OPENAI_API_KEY")
        #     )

        # Testing Params
        self.affordance_mode = "time" # "time", "distance"
        self.dist_threshold = 0.125 # distance threshold for affordance
        self.guide_mode = "trajectory" # "point", "path", "trajectory"
        if self.use_affordance:
            print("############ Affordance Info #############")
            print(f"Using Affordance: {self.use_affordance}")
            print(f"Affordance Duration Mode: {self.affordance_mode}")
            print(f"Affordance Duration: {self.affordance_duration}")
            print(f"Guide Mode: {self.guide_mode}")
            print(f"Distance Threshold: {self.dist_threshold}")
            print("##########################################")

        # load language annotations if using encodings
        self.root_data_dir = "/home/choudhue/PolicyGuide/dataset/calvin_D_fullT_dataset" #root_data_dir #
        # if self.use_lang_encoding:
        self.train_lang_annotations, self.train_ann_to_id, self.train_id_to_actions = self.load_lang_annotations(
            self.root_data_dir + "/training/lang_annotations/auto_lang_ann.npy",
            self.root_data_dir + "/training/trajectories/"
        )
        self.val_lang_annotations, self.val_ann_to_id, self.val_id_to_actions = self.load_lang_annotations(
            self.root_data_dir + "/validation/lang_annotations/auto_lang_ann.npy",
            self.root_data_dir + "/validation/trajectories/"
        )
        # self.train_lang_annotations = self.load_lang_annotations(root_data_dir + "/training/lang_annotations/auto_lang_ann.npy", None)
        # self.val_lang_annotations = self.load_lang_annotations(root_data_dir + "/validation/lang_annotations/auto_lang_ann.npy", None)

        self.phrase_index, self.encoded_instruction_index = self.build_phrase_index(self.train_lang_annotations)

        # use this to plot trajectories
        # self.plot_trajectories_3d()


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        
        if not self._wandb_watch_called:
            wandb.watch(self, log="all", log_freq=100)
            self._wandb_watch_called = True 
        return optimizer

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)

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
            B = batch['robot_obs'].shape[0]  # Batch size
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
            annotations = []
            for idx in batch["language"]:
                annotations.append(self.train_lang_annotations[int(idx[0])])
            encodings = self.language_encoder.encode(annotations)
            batch["observation.embedding"] = encodings
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
        self.guide = None
        self.trajectory_selection = None

    def step(self, obs, goal=None, last_action=None, task_id=None, data_module=None):
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
            converted_obs["observation.embedding"] = self.language_encoder.encode([goal])

        # temporary hardcoded goal
        # goal = "use the switch to turn on the light bulb"
        # "pull the handle to open the drawer"
        # "press the button to turn on the led light"
        # "use the switch to turn on the light bulb"

        padded_guide = None
        out_img = None
        pred_img = {"pred_pixel": None}
        affordance_pixel = None         

        # Use affordance model to get the guide
        if self.use_affordance:
            # Find the euclidean distance between ee_pos and self.guide   
            # TODO: Fix hack! This pulls from the statistics file
            mean = torch.Tensor([0.039233, -0.118554, 0.507826]).cuda(self.cuda_device)
            std = torch.Tensor([0.150769, 0.1104, 0.06253]).cuda(self.cuda_device)
            ee_pos = converted_obs["observation.state"][0, 1, :3] 
            ee_pos = (ee_pos * std) + mean

            # set euc_dist to torch inf
            duration = True
            if self.affordance_mode == "distance":
                euc_dist = torch.inf
                if self.guide is not None:
                    euc_dist = torch.linalg.norm(ee_pos - self.guide)
                    duration = euc_dist > self.dist_threshold
                print(f"Distance to guide: {euc_dist}, Threshold: {self.dist_threshold}")
            elif self.affordance_mode == "time":
                duration = self.rollout_step_counter < self.affordance_duration

            if self.rollout_step_counter % self.replan_freq == 0 and duration: 
                frame = converted_obs["observation.image_static"][0,-1,...].detach().cpu().numpy().copy()
                frame = frame.squeeze()
                frame = ((frame + 1) * 0.5 * 255.0).astype("uint8")
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv2.resize(frame, ([224, 224]))
                affordance_obs = {"img": frame, "lang_goal": goal}
                afford_pred = self.affordance_policy.predict(affordance_obs)
                affordance_pixel = afford_pred["pixel"]
                guide = self.camera.deproject_single_depth(afford_pred["pixel"], afford_pred["depth"])
                self.guide = torch.from_numpy(guide).cuda(self.cuda_device )
                padded_guide = np.concat((guide, last_action[3:]))
                padded_guide = torch.tensor(padded_guide).cuda(self.cuda_device)
                padded_guide = torch.unsqueeze(padded_guide, 0)

                if self.guide_mode == "trajectory":
                    # Generate a trajectory from the guide
                    padded_guide = self.fetch_guide_trajectory(ee_pos, goal, torch.tensor(guide).cuda(self.cuda_device), None) #task_id
                    # padded_guide = self.generate_arc_tensor(ee_pos, torch.tensor(guide).cuda(self.cuda_device), torch.tensor(padded_guide).cuda(self.cuda_device), traj=True)
                    # padded_guide = padded_guide[:8,:]
                elif self.guide_mode == "path":
                    # Generate a path from the guide
                    padded_guide = self.generate_arc_tensor(ee_pos, torch.tensor(guide).cuda(self.cuda_device), None) #torch.tensor(padded_guide).cuda(self.cuda_device)
                    padded_guide = padded_guide[:16,:]
                elif self.guide_mode == "point":
                    # Use the guide as a point
                    padded_guide = padded_guide
                else:
                    raise ValueError(f"Unknown guide mode: {self.guide_mode}")
                
                # Visualize affordance predictions               
                out_img, pred_img = self.affordance_policy.get_preds_viz(affordance_obs, afford_pred)

        # Run inference on diffusion policy
        start = time.perf_counter()
        action, best_idx = self.diffusion_policy.run_inference(converted_obs, guide=padded_guide, guide_mode=self.guide_mode)
        end = time.perf_counter()
        print(f"Run Inference - Execution time: {end - start:.6f} seconds")

        self.rollout_step_counter += 1
        return action, padded_guide, pred_img["pred_pixel"], affordance_pixel, best_idx


    def generate_arc_tensor(self, start, end, guide=None, arc_height=0.0, num_points=16, traj=False):
        """
        Generate a torch.Tensor of shape (num_points, 7) containing 3D arc points
        and 4 extra zero-padding values per point.

        Parameters:
        - start: tuple or list of 3 floats
        - end: tuple or list of 3 floats
        - arc_height: float controlling arc curvature
        - num_points: number of points to generate (default: 8)
        - traj: trajectory guide mode

        Returns:
        - torch.Tensor of shape (num_points, 7)
        """
        start = start.clone()
        end = end.clone()

        mid = (start + end) / 2
        vec = end - start
        dir_vec = vec / vec.norm()

        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).cuda(self.cuda_device)

        # Control point defines the arc
        control = mid + up * arc_height

        # Generate t values (including start and end)
        t_vals = torch.linspace(0, 1, num_points, dtype=torch.float64).cuda(self.cuda_device)

        points = []
        for t in t_vals:
            point = (1 - t)**2 * start + 2 * (1 - t) * t * control + t**2 * end
            padded_point = torch.cat([point, torch.zeros(4).cuda(self.cuda_device)], dim=0)
            points.append(padded_point)

        actions = torch.stack(points)

        if guide is not None:
            #  Combine first 3 position elements of actions to the last 4 Euler Angles of guide
            if traj:
                actions = torch.cat((actions[:, :3], guide[:, 3:]), dim=-1)
            else:
                if guide.shape[0] != actions.shape[0]:
                    guide = guide.repeat(actions.shape[0], 1)
                actions = torch.cat((actions[:, :3], guide[:, 3:]), dim=-1)

        # visualize the generated arc in 3d
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(start[0].cpu().numpy(), start[1].cpu().numpy(), start[2].cpu().numpy(), color='blue', linewidth=2.5, marker='x', markersize=8, label='End Effector Position')
        # ax.plot(end[0].cpu().numpy(), end[1].cpu().numpy(), end[2].cpu().numpy(), color='green', linewidth=2.5, marker='x', markersize=8, label='Affordance Prediction')
        # ax.plot(actions[:, 0].cpu().numpy(), actions[:, 1].cpu().numpy(), actions[:, 2].cpu().numpy(), color='red', linewidth=2.5, marker='x', markersize=5, label='Guide')
        # ax.set_xlabel("X Position")
        # ax.set_ylabel("Y Position")
        # ax.set_zlabel("Z Position")
        # ax.set_title("Generated Trajectories vs. Guide Trajectory")
        # ax.legend()
        # plt.show()
        
        return actions


    def fetch_guide_trajectory(self, current_ee_pos, goal_lang, affordance, task_id=None):
        initial_goal = goal_lang
        
        if self.trajectory_selection is not None:
            print(f"Using previous trajectory selection...")
            # grab index of the current step for all, unless within 8 of the length of trajectory selection, then just use the last 8 steps
            if self.rollout_step_counter * 8 < self.trajectory_selection.shape[0] - 16:
                closest_action = self.trajectory_selection[self.rollout_step_counter * 8:self.rollout_step_counter * 8 + 16, :]
                print(f"Using trajectory selection: ({self.rollout_step_counter * 8}, {self.rollout_step_counter * 8 + 16})")
            else:
                closest_action = self.trajectory_selection[-16:, :]
                print(f"Using trajectory selection: -16")
           
            return closest_action

        if task_id is None:
            goal_lang = self.query_phrase_index(goal_lang, self.phrase_index, self.encoded_instruction_index)

            if goal_lang not in self.train_ann_to_id:
                raise ValueError(f"Goal language '{goal_lang}' not found in training annotations.")
            else:
                skill_id = self.train_ann_to_id[goal_lang]
                actions = self.train_id_to_actions[skill_id]
                print(f"Provided Goal: {initial_goal}, Most Similar Goal: {skill_id}")
        else:
            actions = self.train_id_to_actions[task_id]
            print(f"Provided Goal: {initial_goal}, Provided Task_ID: {task_id}")
        
        min_dist = float('inf')
        closest_action = None
        #send actions to torch cuda
        actions = [torch.tensor(action).cuda(self.cuda_device) for action in actions]
        for action in actions:
            # check if the first action is closest to the end effector position
            euc_dist_first = torch.linalg.norm(current_ee_pos - action[0][:3])
            euc_dist_last = torch.linalg.norm(affordance - action[-1][:3])
            euc_dist = euc_dist_first + euc_dist_last
            if euc_dist < min_dist:
                min_dist = euc_dist
                closest_action = action

        if self.trajectory_selection is None:
            self.trajectory_selection = closest_action

        # store only every eighth action sequence in closest_action
        # indices = torch.linspace(0, 64, steps=8).long()
        # closest_action = closest_action[indices]
        return closest_action[:16, :]


    def load_lang_annotations(self, annotations_path, trajectory_path):
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

        if trajectory_path is not None:
            with gzip.open(trajectory_path + 'ann_to_id.pkl.gz', 'rb') as f:
                ann_to_id = pickle.load(f)

            with gzip.open(trajectory_path + 'id_to_actions.pkl.gz', 'rb') as f:
                id_to_actions = pickle.load(f)

            return lang_annotations, ann_to_id, id_to_actions
        return lang_annotations, None, None


    def build_phrase_index(self, annotation_dict: list) -> torch.Tensor:
        """
        Encode a list of phrases using the language encoder.

        Args:
            phrases (list): List of phrases to encode.

        Returns:
            torch.Tensor: Encoded phrases.
        """
        # if not self.use_lang_encoding:
        #     raise ValueError("Language encoding is not enabled.")
        
        # build the phrase index from the annotation dictionary
        phrases = list(set(annotation_dict.values()))

        encodings = self.language_encoder.encode(phrases, self.normalize_language_embeddings)

        # encodings = []
        # for phrase in phrases:
        #     encoding = self.language_encoder.encode_text(phrase)[0]
        #     encodings.append(encoding)

        # encodings = torch.stack(encodings).squeeze().cuda()

        return phrases, encodings

    def plot_trajectories_3d(self, max_trajs_per_task=30, single_task="rotate_blue_block_right"):
        """
        Plots 3D trajectories from self.val_id_to_actions.
        Each task ID has up to `max_trajs_per_task` trajectories plotted and labeled once in the legend.
        
        Assumes: self.val_id_to_actions[task_id].shape == (100, 65, 7)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Generate a distinct color for each task using a colormap
        task_ids = list(self.val_id_to_actions.keys())
        colormap = cm.get_cmap('tab20', len(task_ids))  # Or 'tab20', 'hsv', etc.
        task_to_color = {task_id: colormap(i) for i, task_id in enumerate(task_ids)}

        for task_id, traj_batch in self.val_id_to_actions.items():
            traj_list = [np.array(action) for action in traj_batch]

            if not isinstance(traj_list, (list, tuple)):
                print(f"Task {task_id} is not a list, skipping.")
                continue

            if single_task is not None and task_id != single_task:
                print(f"Task {task_id} is not {single_task}, skipping.")
                continue

            # skills = ["open_drawer", "move_slider_left", "close_drawer", "turn_on_lightbulb", "turn_off_lightbulb", "move_slider_right", "turn_on_led", "turn_off_led", "lift_blue_block_drawer", "lift_red_block_drawer", "lift_pink_block_drawer", "lift_blue_block_table", "lift_red_block_table", "lift_blue_block_slider", "lift_red_block_slider", "lift_pink_block_slider", "unstack_block"]
            # if task_id not in skills:
            #     print(f"Task {task_id} is not {single_task}, skipping.")
            #     continue

            count = 0
            for i, traj in enumerate(traj_list):
                if not isinstance(traj, np.ndarray) or traj.ndim != 2 or traj.shape[1] != 7:
                    print(f"Skipping trajectory {i} in task {task_id} due to shape: {traj.shape if isinstance(traj, np.ndarray) else 'Invalid type'}")
                    continue

                x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
                color = task_to_color[task_id]
                label = f"Task {task_id}" if count == 0 else None  # Only label once per task

                ax.plot(x, y, z, label=label, color=color, alpha=0.7)
                count += 1

                if count >= max_trajs_per_task:
                    break

        if single_task is not None and task_id != single_task:
            ax.set_title(f"3D Trajectories for {single_task} Task")
        else:
            ax.set_title("3D Trajectories for All Tasks")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.5, 0.3])
        ax.set_zlim([0.1, 0.9])

        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def query_phrase_index(self, phrase, phrases, encodings):
        """
        Query the phrase index for the closest phrase to the given phrase.

        Args:
            phrase (str): The phrase to query.
            phrases (list): List of phrases in the index.
            encodings (torch.Tensor): Encoded phrases in the index.

        Returns:
            str: The closest phrase in the index.
        """
        # if not self.use_lang_encoding:
        #     raise ValueError("Language encoding is not enabled.")
        # if self.use_chatgpt:
        #     # Use ChatGPT to describe the phrase
        #     gpt_output = self.chatgpt_describer.describe([phrase])
        #     phrase = gpt_output.three_words

        query_encoding = self.language_encoder.encode([phrase], self.normalize_language_embeddings)

        # query_encoding = self.language_encoder.encode_text(phrase)[0]
        similarities = torch.cosine_similarity(encodings, query_encoding[0].squeeze(0), dim=-1)
        
        # similarities = util.dot_score(encodings, query_encoding)  # assumes encodings is shape (N, D), query_encoding is (1, D)
        # similarities = similarities.squeeze(0)  # remove batch dim
        closest_idx = torch.argmax(similarities).item()
        
        return phrases[closest_idx]
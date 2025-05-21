import argparse
from collections import Counter, defaultdict, deque
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import imageio

# This is for using the locally installed repo clone when using slurm
from itpg.policy.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from itpg.evaluation.multistep_sequences import get_sequences
from itpg.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
    visualize_point,
    remove_oldest_sphere,
    visualize_point_policy,
)
from itpg.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import pybullet as p

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 45  # 8 actions step
NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)
    print(env.p)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def save_images_and_create_gif(images: List[np.ndarray], save_dir: str, gif_name: str = "rollout.gif", fps: int = 10):
    """
    Save images from observations and create a GIF.

    Args:
        images (List[np.ndarray]): List of images (numpy arrays) to save and include in the GIF.
        save_dir (str): Directory to save the images and GIF.
        gif_name (str): Name of the output GIF file.
        fps (int): Frames per second for the GIF.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save individual images
    image_paths = []
    for idx, img in enumerate(images):
        img_path = save_path / f"frame_{idx:04d}.png"
        imageio.imwrite(img_path, img)
        image_paths.append(img_path)

    # Create GIF
    gif_path = save_path / gif_name
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for img_path in image_paths:
            writer.append_data(imageio.v2.imread(img_path))
    
    for img_path in image_paths:
        os.remove(img_path)

    print(f"GIF saved at {gif_path}")


def combine_gifs(folder_path: str, output_gif_name: str = "combined_sequences.gif", fps: int = 30):
    """
    Combine multiple GIFs into a single GIF.

    Args:
        folder_path (str): Path to the folder containing the GIFs.
        output_gif_name (str): Name of the output combined GIF file.
        fps (int): Frames per second for the combined GIF.
    """
    gif_files = sorted(Path(folder_path).glob("*.gif"))
    images = []

    for gif_file in gif_files:
        with imageio.get_reader(gif_file) as reader:
            for frame in reader:
                images.append(frame)

    output_gif_path = Path(folder_path) / output_gif_name
    with imageio.get_writer(output_gif_path, mode="I", fps=fps) as writer:
        for img in images:
            writer.append_data(img)

    print(f"Combined GIF saved at {output_gif_path}")


class CustomModel(CalvinBaseModel):
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False, save_viz=False, viz_folder=None, curr_time=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    # task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/calvin_D_3T_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    # val_annotations = OmegaConf.load(conf_dir / "annotations/calvin_D_3T_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    # Temporary hardcoded sequences for testing
    # eval_sequences = [({'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_right', 'pink_block': 'slider_left', 'grasped': 0}, (('turn_on_lightbulb', 'open_drawer', 'turn_on_led'))),
    #                   ({'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'closed', 'red_block': 'slider_right', 'blue_block': 'slider_left', 'pink_block': 'table', 'grasped': 0}, (('open_drawer', 'turn_on_led', 'turn_on_lightbulb'))),
    #                   ({'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_left', 'pink_block': 'table', 'grasped': 0}, (('turn_on_led', 'turn_on_lightbulb', 'open_drawer')))]
    
    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        with torch.amp.autocast('cuda'):
            result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, save_viz, viz_folder, curr_time)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, save_viz, viz_folder, curr_time):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, save_viz, viz_folder, curr_time)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def combine_observations(observations: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Combine an unspecified number of observations for each of their keys.

    Args:
        observations (list): List of observation dictionaries.

    Returns:
        dict: Combined observation dictionary.
    """
    merged = {}
    merged['rgb_obs'] = {}

    merged['robot_obs'] = torch.cat([obs['robot_obs'] for obs in observations], dim=1)
    merged['rgb_obs']['rgb_static'] = torch.cat([obs['rgb_obs']['rgb_static'] for obs in observations], dim=1)
    merged['rgb_obs']['rgb_gripper'] = torch.cat([obs['rgb_obs']['rgb_gripper'] for obs in observations], dim=1)
    # print(f"Robot_obs shape: {merged['robot_obs'].shape}")
    # print(f"Image Shape {merged['rgb_obs']['rgb_static'].shape}")

    return merged


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, save_viz, viz_folder, curr_time):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask}")
        time.sleep(0.5)
    obs = env.get_obs()
    guide_viz = deque()
    action_viz = deque()
    client_id = env.cid  # or env.sim.physics_client
    # print(f"Calvin Physics Client ID: {client_id}")
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    obs_history = None
    if save_viz and debug:
        images = []

    last_action = np.zeros((7))
    for step in tqdm(range(EP_LEN)):
        if obs_history is None:
            # If there is no past observation, use the current observation twice
            obs_history = [obs, obs]
                        
        combined_obs = combine_observations(obs_history)
        guide = None
        
        # action = model.step(combined_obs, lang_annotation)
        action, guide = model.step(combined_obs, lang_annotation, last_action)

        # save last action for padding
        last_action = action[:, -1, :].squeeze().cpu().numpy()
        # trajectory_pts = []

        # for i in range(action.shape[1]):
        #     # action_viz.append(visualize_point(client_id, action[:, i, :3].squeeze(), i))
        #     trajectory_pts.append(action[:, i, :3].squeeze().tolist())
        #     # if step >= 1:
        #     #     remove_oldest_sphere(action_viz, client_id)

        # Visualize the trajectory using the deque of points
        # cylinder_ids, cylinder_colors = visualize_point_policy(client_id, action[:, :, :3].squeeze())
        # print(cylinder_ids, cylinder_colors)

        # if debug:
        #     if guide is not None:
        #         guide_viz.append(visualize_point(client_id, guide[0], 7))

        for i in range(action.shape[1]):
            obs, _, _, current_info = env.step(action[:,i,...])
            # print(f'Observation: {obs["rgb_obs"]["rgb_static"].shape},\n') #Action: {action[:,i,...]}\n\n")
            obs_history = obs_history[-1:]
            obs_history.append(obs)        

            if debug:
                # for cylinder_id, color in zip(cylinder_ids, cylinder_colors):
                #     # Keep the original RGB values, but set alpha to 1.0 for full opacity
                #     p.changeVisualShape(
                #         objectUniqueId=cylinder_id,
                #         linkIndex=-1,
                #         rgbaColor=[color[0], color[1], color[2], 1.0],  # Update only the alpha
                #         physicsClientId=client_id
                #     )
                if save_viz:
                    temp_obs = (obs["rgb_obs"]["rgb_static"][:,0,...] * 255).clamp(0, 255).byte().squeeze()
                    temp_obs = temp_obs.squeeze().permute(1, 2, 0).cpu().numpy()
                    images.append(temp_obs)
                else:
                    img = env.render(mode="rgb_array")
                    # print(f"Image shape: {img.shape}")
                    join_vis_lang(img, lang_annotation)
                # for cylinder_id, color in zip(cylinder_ids, cylinder_colors):
                #     # Keep the original RGB values, but set alpha to 0.0
                #     p.changeVisualShape(
                #         objectUniqueId=cylinder_id,
                #         linkIndex=-1,
                #         rgbaColor=[color[0], color[1], color[2], 0.0],  # Update only the alpha
                #         physicsClientId=client_id
                #     )
                # time.sleep(0.1)
            # if step == 0:
            #     # for tsne plot, only if available
            #     collect_plan(model, plans, subtask)

            # if step >= 10:
            #     if guide is not None:
            #         remove_oldest_sphere(guide_viz, client_id)

        if guide is not None and guide_viz:
            remove_oldest_sphere(guide_viz, client_id)

        # for cylinder_id in cylinder_ids:
        #     p.removeBody(cylinder_id, physicsClientId=client_id)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("task success", "green"), end=" \n")
                if save_viz:
                    # save images for gif
                    save_dir = os.path.join(viz_folder, f"eval_viz_{curr_time}")
                    sequence_time = time.strftime("%Y%m%d_%H%M%S")
                    gif_name = f"{sequence_time}_rollout_{subtask}_success.gif"
                    save_images_and_create_gif(images, save_dir, gif_name)
            return True
    if debug:
        print(colored("task failed", "red"), end=" \n")
        if save_viz:
            # save images for gif
            save_dir = os.path.join(viz_folder, f"eval_viz_{curr_time}")
            sequence_time = time.strftime("%Y%m%d_%H%M%S")
            gif_name = f"{sequence_time}_rollout_{subtask}_fail.gif"
            save_images_and_create_gif(images, save_dir, gif_name)
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--save_viz", action="store_true", help="Save visualization of environment")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    curr_time = time.strftime("%Y%m%d_%H%M%S")

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug, save_viz=args.save_viz)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]
        print(checkpoints)
        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, 
                            env, 
                            epoch, 
                            eval_log_dir=args.eval_log_dir, 
                            debug=args.debug, 
                            create_plan_tsne=False, 
                            save_viz=args.save_viz, 
                            viz_folder=args.train_folder, 
                            curr_time=curr_time
                        )


if __name__ == "__main__":
    main()

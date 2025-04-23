from collections import Counter, deque
import contextlib
import json
import logging
import os
from pathlib import Path
from pydoc import locate

from itpg.policy.models.itpg import ITPG
from itpg.utils.utils import add_text, format_sftp_path
import cv2
import hydra
import numpy as np
from numpy import pi
from omegaconf import OmegaConf
import pyhash
import torch
import pybullet
from scipy.interpolate import splprep, splev

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    print(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    # datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    datasets_cfg = hydra.compose("policy_guide_dataset.yaml", overrides=["lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset #.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    # import the model class that was used for the training
    model_cls = locate(cfg.model._target_)
    model = model_cls.load_from_checkpoint(checkpoint)
    model.load_lang_embeddings(dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy")
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module


def collect_plan(model, plans, subtask):
    try:
        plans[subtask].append((model.plan.cpu(), model.latent_goal.cpu()))
    except AttributeError:
        return


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img['rgb_static']
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, log_dir, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def create_tsne(plan_dict, log_dir, epoch):
    ids, labels, plans, latent_goals = zip(
        *[
            (i, label, latent_goal, plan)
            for i, (label, plan_list) in enumerate(plan_dict.items())
            for latent_goal, plan in plan_list
        ]
    )
    latent_goals = torch.cat(latent_goals)
    plans = torch.cat(plans)
    np.savez(f"{log_dir / f'tsne_data_{epoch}.npz'}", ids=ids, labels=labels, plans=plans, latent_goals=latent_goals)


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def interpolate_gradient(base_gradient, num_segments):
    """
    Interpolates a gradient to match the number of segments.

    Args:
        base_gradient: A list of RGBA colors defining the base gradient.
        num_segments: The number of segments to generate colors for.

    Returns:
        A list of RGBA colors interpolated to match the number of segments.
    """
    base_gradient = np.array(base_gradient)
    base_positions = np.linspace(0, 1, len(base_gradient))  # Positions of base colors
    target_positions = np.linspace(0, 1, num_segments)  # Positions for interpolated colors

    # Interpolate each channel (R, G, B, A) separately
    interpolated_gradient = np.zeros((num_segments, 4))
    for i in range(4):  # Iterate over RGBA channels
        interpolated_gradient[:, i] = np.interp(target_positions, base_positions, base_gradient[:, i])

    return interpolated_gradient.tolist()


def visualize_point_policy(client_id, points, smooth_factor=20, gradient=None, line_width=0.005):
    """
    Visualizes the action trajectory as a series of transparent cylinders with spline smoothing.

    Args:
        client_id: The PyBullet client ID.
        points: A list of 3D points representing the trajectory.
        gradient: A list of RGBA colors for the gradient. Defaults to a rainbow gradient.
        line_width: The radius of the cylinders.
        smooth_factor: Number of interpolated points between each pair of original points.
    Returns:
        A tuple containing:
            - A list of cylinder IDs for further manipulation.
            - A list of RGBA colors used for the cylinders.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    elif not isinstance(points, np.ndarray):
        raise ValueError("Points must be a torch tensor or numpy array.")

    # Interpolate the points using a spline
    tck, u = splprep(points.T, s=0)  # Create a spline representation
    u_fine = np.linspace(0, 1, len(points) * smooth_factor)  # Generate fine-grained parameter values
    smooth_points = np.array(splev(u_fine, tck)).T  # Evaluate the spline to get smooth points

    if gradient is None:
        # Define a default rainbow gradient
        gradient = [
            [1.0, 0.0, 0.0, 0.1],  # Red (transparent)
            [1.0, 0.5, 0.0, 0.1],  # Orange (transparent)
            [1.0, 1.0, 0.0, 0.1],  # Yellow (transparent)
            [0.0, 1.0, 0.0, 0.1],  # Green (transparent)
            [0.0, 0.0, 1.0, 0.1],  # Blue (transparent)
            [0.29, 0.0, 0.51, 0.1],  # Indigo (transparent)
            [0.56, 0.0, 1.0, 0.1],  # Violet (transparent)
        ]

    num_segments = smooth_points.shape[0] - 1
    
    # Interpolate the gradient to match the number of segments
    gradient = interpolate_gradient(gradient, num_segments)

    cylinder_ids = []
    cylinder_colors = []

    for i in range(num_segments):
        start_point = smooth_points[i]
        end_point = smooth_points[i + 1]
        color = gradient[i % len(gradient)]  # Cycle through the gradient colors

        midpoint = (start_point + end_point) / 2
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction = direction / length  # Normalize the direction vector

        # Compute orientation for the cylinder
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        if np.linalg.norm(rotation_axis) < 1e-6:
            orientation = pybullet.getQuaternionFromEuler([0, 0, 0]) if direction[2] > 0 else pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            orientation = pybullet.getQuaternionFromAxisAngle(rotation_axis.tolist(), rotation_angle)

        # Create a visual shape for the cylinder
        visual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_CYLINDER,
            radius=line_width,
            length=length,
            rgbaColor=color,
            physicsClientId=client_id
        )

        # Create a multi-body to place the cylinder in the scene
        cylinder_id = pybullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=midpoint.tolist(),
            baseOrientation=orientation,
            physicsClientId=client_id
        )

        cylinder_ids.append(cylinder_id)
        cylinder_colors.append(color)  # Store the original color

    return cylinder_ids, cylinder_colors


def visualize_point(client_id, point, color_idx=8):
    gradient = [
        [0.2, 0.0, 0.0, 1.0],  # Very dark red
        [0.35, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 1.0],
        [0.65, 0.0, 0.0, 1.0],
        [0.75, 0.0, 0.0, 1.0],
        [0.85, 0.0, 0.0, 1.0],
        [0.95, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],   # Bright red
        [0.2, 0.2, 1.0, 1.0],   # Blue
    ]
    # Define sphere properties
    radius = 0.02
    mass = 0  # Use 0 for a static sphere
    position = [point[0], point[1], point[2]]  # Your desired 3D point

    # Create sphere in the Calvin environment's physics client
    collision_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
    visual_shape = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=radius, rgbaColor=gradient[color_idx], physicsClientId=client_id)

    # Create the actual sphere body
    sphere_id = pybullet.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id
    )

    # print(f"Sphere added at {position} with ID {sphere_id}")
    return sphere_id

def remove_oldest_sphere(queue, client_id):
    if queue:
        sphere_id = queue.popleft()  # Get and remove the oldest sphere
        pybullet.removeBody(sphere_id, physicsClientId=client_id)
        # print(f"Removed sphere with ID {sphere_id}")
    else:
        print("No spheres left to remove.")


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        f"Average Success Rate {mod} = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs

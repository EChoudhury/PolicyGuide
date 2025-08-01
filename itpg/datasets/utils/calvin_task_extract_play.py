if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import datetime
import json
from itpg.datasets.utils.robot_replay_buffer import RobotReplayBuffer
# import torch
import random


logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class CALVINSkillExtractor:
    """
    This class is used to extract skills from the raw CALVIN dataset.
    An object of this is iterable and returns one episode of the chosen skill
    as a dictionary.
    """

    def __init__(
        self,
        data_dir: str,
        save_dir: str,
        data_to_extract: list,
        step_len: int,
        n_episodes: int = 3000,
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.data_to_extract = data_to_extract
        self.n_episodes = n_episodes
        self.max_window_size = 32
        self.min_window_size = 16
        self.episode_lookup, self.annotations_idx = self.load_file_indices(self.data_dir)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        self.step_len = step_len
        self.replay_buffer = RobotReplayBuffer.create_from_path(self.save_dir, mode="a")
        
    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        return self.get_sequences(idx)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [
            load_npz(self.get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episode = {
            key: np.stack([ep[key] for ep in episodes])
            for key, _ in episodes[0].items()
        }
        return episode

    def get_sequences(self, idx: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + self.max_window_size

        episode = self.zip_sequence(start_file_indx, end_file_indx)

        batch = {}
        if "robot_obs" in self.data_to_extract:
            batch.update({"robot_obs": episode["robot_obs"]})

        if "scene_obs" in self.data_to_extract:
            batch.update({"scene_obs": episode["scene_obs"]})

        if "actions" in self.data_to_extract:
            batch.update({"actions": episode["actions"]})

        # Note: ToTensor() automatically normalizes images (uint8 or byte arrays) to [0, 1] range
        # To avoid that, one could change image type to np.int64 before calling ToTensor()
        # See: https://discuss.pytorch.org/t/does-pytorch-automatically-normalizes-image-to-0-1/40022/2
        if "rgb_gripper" in self.data_to_extract:
            batch.update({"rgb_gripper": episode["rgb_gripper"].astype(np.int64)})

        if "rgb_static" in self.data_to_extract:
            batch.update({"rgb_static": episode["rgb_static"].astype(np.int64)})

        return batch
        

    def load_file_indices(self, abs_datasets_dir: Path) -> List:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        # print(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        # for start_idx, end_idx in ep_start_end_ids:
        #     assert end_idx > self.max_window_size
        start_idx = 53819
        end_idx = 611098

        for idx in range(start_idx, end_idx + 1 - self.max_window_size, self.max_window_size):
            episode_lookup.append(idx)
            if len(episode_lookup) >= self.n_episodes:
                print(f"Loaded {len(episode_lookup)} episodes from {abs_datasets_dir}.")
                return episode_lookup, None
        print(f"Loaded all {len(episode_lookup)} episodes from {abs_datasets_dir}.")
        return episode_lookup, None


def generate_episode_dict(episode):
    eps_len = int(episode["robot_obs"].shape[0])

    episode_dict = []
    for i in range(eps_len):
        episode_dict.append({
            "robot_obs": episode["robot_obs"][i],
            "rgb_static": episode["rgb_static"][i],
            "rgb_gripper": episode["rgb_gripper"][i],
            "actions": episode["actions"][i],
            "episode_step": i,
        })

    return episode_dict

def make_dataset(load_path, save_dir, step_len, n_episodes=2000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    skill_list = [
        "open_drawer",
        # "move_slider_left",
        # "lift_pink_block_table",
        # "push_pink_block_right",
        # "close_drawer",
        # "turn_on_lightbulb",
        # "turn_off_lightbulb",
        # "move_slider_right",
        # "turn_on_led",
        # "turn_off_led",
        # "lift_blue_block_drawer",
        # "lift_red_block_drawer",
        # "lift_pink_block_drawer",
        # "lift_blue_block_table",
        # "lift_red_block_table",
        # "lift_blue_block_slider",
        # "lift_red_block_slider",
        # "lift_pink_block_slider",
        # "push_blue_block_left",
        # "push_red_block_left",
        # "push_pink_block_left",
        # "push_blue_block_right",
        # "push_red_block_right",
        # "rotate_red_block_right",
        # "rotate_red_block_left",
        # "rotate_blue_block_right",
        # "rotate_blue_block_left",
        # "rotate_pink_block_right",
        # "rotate_pink_block_left",
        # "place_in_slider",
        # "place_in_drawer",
        # "stack_block",
        # "unstack_block",
        # "push_into_drawer",
    ]
    data_to_extract = [
        "robot_obs",
        "episode_step",
        "actions",
        "rgb_static",
        "rgb_gripper",
    ]

    info = {
        "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data": data_to_extract,
        "skills": skill_list,
    }

    logger.info(f"Extracting data...")

    extractor = CALVINSkillExtractor(
        data_dir=load_path,
        save_dir=save_dir,
        data_to_extract=data_to_extract,
        step_len=step_len,
        n_episodes=n_episodes,
    )

    for idx in tqdm(range(len(extractor))):
        episode = extractor[idx]

        episode_dict = generate_episode_dict(episode)

        extractor.replay_buffer.add_episode_from_list(episode_dict, compressors="disk") #chunks=desired_chunks,
        # print(f"Saving episode for {skill}...")
        # np.savez(
        #     os.path.join(save_dir, f"{skill}.npz"),
        #     states=states,
        #     actions=actions,
        #     traj_lengths=traj_lengths.astype(int),
        #     rgb_statics=rgb_statics,
        #     rgb_grippers=rgb_grippers,
        # )

    path = os.path.join(save_dir, "info.json")
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            json.dump(info, f)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        default="/home/choudhue/PolicyGuide/dataset/task_D_D/calvin_d_dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/choudhue/PolicyGuide/dataset/calvin_D_full_play_dataset",
    )
    parser.add_argument("--step_len", type=int, default=1)
    parser.add_argument("--full", help="Use this flag to load both training and validation data.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_episodes_train", type=int, default=600000, help="Number of episodes to extract.")
    parser.add_argument("--n_episodes_val", type=int, default=10, help="Number of episodes to extract.")
    args = parser.parse_args()

    print(args)

    if args.full:
        # Load training data
        load_path = os.path.join(args.load_path, "training")
        save_dir = os.path.join(args.save_dir, "training")
        make_dataset(load_path, save_dir, args.step_len, n_episodes=args.n_episodes_train)
        # Load validation data
        load_path = os.path.join(args.load_path, "validation")
        save_dir = os.path.join(args.save_dir, "validation")
        make_dataset(load_path, save_dir, args.step_len, n_episodes=args.n_episodes_val)
    else:
        make_dataset(args.load_path, args.save_dir, args.step_len, n_episodes=args.n_episodes_train)
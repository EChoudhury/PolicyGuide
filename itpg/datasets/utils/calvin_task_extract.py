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
        skill_name: str,
        data_to_extract: list,
        step_len: int,
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.skill_name = skill_name
        self.data_to_extract = data_to_extract
        self.episode_lookup, self.annotations_idx = self.load_file_indices(self.data_dir, self.skill_name)
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
            for file_idx in range(start_idx, end_idx + 1, self.step_len)
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
        info_indx = self.episode_lookup[idx]
        start_file_indx = info_indx[0]
        end_file_indx = info_indx[1] + 32

        lang_annotation = self.annotations_idx[idx]

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

        if "language" in self.data_to_extract:
            batch.update({"language": lang_annotation})

        return batch

    def load_file_indices(self, data_dir: Path, skill: str) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes
        parameters
        ----------
        data_dir:               absolute path of the directory containing the datasets
        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert data_dir.is_dir()
        skill_name = skill

        episode_lookup = []

        file_name = data_dir / "lang_annotations" / "auto_lang_ann.npy"
        data = np.load(file_name, allow_pickle=True).reshape(-1)[0]

        all_eps_idx_part_task = [
            i for (i, v) in enumerate(data["language"]["task"]) if v == skill_name
        ]
        # all_eps_idx_annotations = [
        #     data["language"]["ann"][i] for i in all_eps_idx_part_task
        # ]
        all_eps_start_end_part_task = [
            data["info"]["indx"][i] for i in all_eps_idx_part_task
        ]

        for i in range(len(all_eps_start_end_part_task)):
            episode_lookup.append(all_eps_start_end_part_task[i])

        logger.info(
            f"Found {len(episode_lookup)} demonstrations of skill {skill_name}."
        )
        return episode_lookup, all_eps_idx_part_task

def generate_episode_dict(episode, language):
    eps_len = int(episode["robot_obs"].shape[0])

    # Low-dim observations (robot_no_joints)
    # selected_obs = list(range(0, 7)) + [14]
    # state_obs = episode["robot_obs"][:, selected_obs]

    episode_dict = []
    for i in range(eps_len):
        episode_dict.append({
            "language": language,
            "robot_obs": episode["robot_obs"][i],
            "rgb_static": episode["rgb_static"][i],
            "rgb_gripper": episode["rgb_gripper"][i],
            "actions": episode["actions"][i],
            "episode_step": i,
        })

    return episode_dict

def make_dataset(load_path, save_dir, step_len, multi_dir=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    skill_list = [
        # "open_drawer",
        # "move_slider_left",
        # "lift_pink_block_table",
        # "push_pink_block_right",
        # "close_drawer",
        "turn_on_lightbulb",
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
        "language",
    ]

    info = {
        "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data": data_to_extract,
        "skills": skill_list,
    }

    for skill in tqdm(skill_list, disable=True):
        logger.info(f"Extracting data for skill: {skill}")

        if multi_dir:
            dir = os.path.join(save_dir, skill)
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
        else:
            dir = save_dir

        extractor = CALVINSkillExtractor(
            data_dir=load_path,
            save_dir=dir,
            skill_name=skill,
            data_to_extract=data_to_extract,
            step_len=step_len,
        )

        for idx in tqdm(range(len(extractor))):
            episode = extractor[idx]
            
            episode_dict = generate_episode_dict(episode, extractor.annotations_idx[idx])

            extractor.replay_buffer.add_episode_from_list(episode_dict, compressors="disk")
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
        default="/home/choudhue/PolicyGuide/dataset/calvin_D_1T_dataset",
    )
    parser.add_argument("--step_len", type=int, default=1)
    parser.add_argument("--full", help="Use this flag to load both training and validation data.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multi", help="Use this flag to create a separate folder for each task.", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print(args)

    if args.full:
        # Load training data
        load_path = os.path.join(args.load_path, "training")
        save_dir = os.path.join(args.save_dir, "training")
        make_dataset(load_path, save_dir, args.step_len, args.multi)
        # Load validation data
        load_path = os.path.join(args.load_path, "validation")
        save_dir = os.path.join(args.save_dir, "validation")
        make_dataset(load_path, save_dir, args.step_len, args.multi)
    else:
        make_dataset(args.load_path, args.save_dir, args.step_len, args.multi)
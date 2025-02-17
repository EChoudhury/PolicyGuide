from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from pathlib import Path
import numpy as np
import torch
from itpg.policy.models.diffusion_policy.utils.normalization import create_stats_buffers

def compute_statistics(data_path, data_keys):
    stats = {key: {"mean": 0, "std": 0, "min": np.inf, "max": -np.inf, "count": 0} for key in data_keys}
    indices = next(iter(np.load(f"{data_path}/scene_info.npy", allow_pickle=True).item().values()))
    indices = list(range(indices[0], indices[1] + 1))

    for idx in indices:
        t = np.load(f"{data_path}/episode_{idx:07d}.npz", allow_pickle=True)
        for key in data_keys:
            if key in t:
                data = t[key]
                count = data.shape[0] * data.shape[1]
                stats[key]["count"] += count

                # Update mean
                delta = data - stats[key]["mean"]
                stats[key]["mean"] += delta.sum(axis=(0, 1)) / stats[key]["count"]

                # Update std
                delta2 = data - stats[key]["mean"]
                stats[key]["std"] += (delta * delta2).sum(axis=(0, 1))

                # Update min and max
                stats[key]["min"] = np.minimum(stats[key]["min"], data.min(axis=(0, 1)))
                stats[key]["max"] = np.maximum(stats[key]["max"], data.max(axis=(0, 1)))

    for key in data_keys:
        stats[key]["std"] = np.sqrt(stats[key]["std"] / stats[key]["count"])

    return stats
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument("path", type=str, default="/home/choudhue/PolicyGuide/dataset/", help="Path to dir containing scene_info.npy")
    parser.add_argument("-d", "--data", nargs="*", default=["rgb_static", "rgb_gripper", "robot_obs", "actions"], help="Data to visualize")
    parser.add_argument("-s", "--save", type=str, default="/home/choudhue/PolicyGuide/dataset/stats/stats_buffers.pth", help="Path to save stats buffers")
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        print(f"Path {args.path} is either not a directory, or does not exist.")
        exit()

    stats = compute_statistics(args.path, args.data)

    shapes = {key: list(stats[key]["mean"].shape) for key in args.data}
    modes = {key: "mean_std" if key in ["rgb_gripper", "rgb_static"] else "min_max" for key in args.data}
    stats_tensors = {key: {stat: torch.tensor(value) for stat, value in stats[key].items()} for key in args.data}

    stats_buffers = create_stats_buffers(shapes, modes, stats_tensors)
    torch.save(stats_buffers, args.save)
    print(f"Stats buffers saved to {args.save}")
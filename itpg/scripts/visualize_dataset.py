from argparse import ArgumentParser
from pathlib import Path
import pickle
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

from pathlib import Path
import numpy as np
import torch
from itpg.policy.models.diffusion_policy.configuration_diffusion import DiffusionConfig
from itpg.policy.models.diffusion_policy.utils.normalization import create_stats_buffers

def compute_statistics(data_path, data_keys):
    stats = {key: {"mean": np.zeros(3), "M2": np.zeros(3), "std": np.zeros(3), "min": np.inf, "max": -np.inf, "count": 0, "hist": np.zeros((3, 256))} for key in data_keys}
    indices = next(iter(np.load(f"{data_path}/scene_info.npy", allow_pickle=True).item().values()))
    indices = list(range(indices[0], indices[1] + 1))

    for idx in tqdm.tqdm(indices, desc="Processing episodes"):
        t = np.load(f"{data_path}/episode_{idx:07d}.npz", allow_pickle=True)
        for key in data_keys:
            if key in t:
                data = t[key]
                if key in ["rgb_static", "rgb_gripper"]:
                    count = data.shape[0] * data.shape[1]
                    stats[key]["count"] += count

                    # Welford's online algorithm for mean and variance
                    delta = data - stats[key]["mean"]
                    stats[key]["mean"] += delta.sum(axis=(0, 1)) / stats[key]["count"]
                    delta2 = data - stats[key]["mean"]
                    stats[key]["M2"] += (delta * delta2).sum(axis=(0, 1))

                    # Update min and max
                    stats[key]["min"] = np.minimum(stats[key]["min"], data.min(axis=(0, 1)))
                    stats[key]["max"] = np.maximum(stats[key]["max"], data.max(axis=(0, 1)))
                    
                    # Compute histogram
                    # for channel in range(3):
                    #     stats[key]["hist"][channel] += np.histogram(data[..., channel], bins=256, range=(0, 255))[0]
                else:
                    stats[key]["min"] = np.minimum(stats[key]["min"], data.min())
                    stats[key]["max"] = np.maximum(stats[key]["max"], data.max())

    for key in data_keys:
        if key in ["rgb_static", "rgb_gripper"]:
            stats[key]["std"] = np.sqrt(stats[key]["M2"] / (stats[key]["count"] - 1))  # Unbiased std estimate
    
    return stats

def format_stats(stats):
    formatted_stats = {}
    for key, value in stats.items():
        if key=="rgb_static":  
            key = "observation.image_static"
        if key=="rgb_gripper":  
            key = "observation.image_wrist"
        if key=="robot_obs":  
            key = "observation.state"
        if key=="actions":  
            key = "action"
        formatted_stats[key] = {
            "mean": torch.tensor(value["mean"], dtype=torch.float32),
            "std": torch.tensor(value["std"], dtype=torch.float32),
            "min": torch.tensor(value["min"], dtype=torch.float32),
            "max": torch.tensor(value["max"], dtype=torch.float32),
            # "hist": torch.tensor(value["hist"], dtype=torch.float32),
        }
    return formatted_stats

def plot_histogram(stats):
    for key in ["rgb_static", "rgb_gripper"]:
        if key in stats:
            plt.figure(figsize=(12, 4))
            for channel, color in enumerate(["red", "green", "blue"]):
                plt.plot(stats[key]["hist"][channel], color=color, label=f"{key} {color}")
            plt.title(f"Histogram for {key}")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute statistics of dataset")
    parser.add_argument("--path", type=str, default="/home/choudhue/PolicyGuide/dataset/calvin_debug_dataset", help="Path to dir containing scene_info.npy")
    parser.add_argument("-d", "--data", nargs="*", default=["rgb_static", "rgb_gripper", "robot_obs", "actions"], help="Data to compute statistics for")
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        print(f"Path {args.path} is either not a directory, or does not exist.")
        exit()

    train_path = os.path.join(args.path, 'training') 
    stats = compute_statistics(train_path, args.data)
    formatted_stats = format_stats(stats)
    print(formatted_stats)

    #TODO: Incorporate validation data?
    
    data_file = os.path.join(args.path, 'stats') 
    os.makedirs(data_file, exist_ok=True) 
    data_file = os.path.join(data_file, 'dataset.pkl') 
    with open(data_file, 'wb') as f:
        pickle.dump(formatted_stats, f)
    print(f"Stats saved to {data_file}")

    # plot_histogram(formatted_stats)

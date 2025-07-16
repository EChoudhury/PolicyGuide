import argparse
import os
from pathlib import Path
from typing import List
import cv2
import imageio.v2 as imageio
import numpy as np
from itpg.datasets.policy_guide_dataset import PolicyGuideDataset
from itpg.datasets.policy_guide_data_module import PolicyGuideDataModule
from omegaconf import OmegaConf, DictConfig
import hydra
import time
import matplotlib.pyplot as plt
from lovely_numpy import lovely
import re
import pickle
import tqdm

@hydra.main(config_path="/home/choudhue/PolicyGuide/conf/datamodule", config_name="default")
def visualize_dataset(cfg: DictConfig, gui=True, analyze=True) -> None:
    # Path to the Zarr dataset
    zarr_path = "/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset"

    # Initialize the PolicyGuideDataset
    print("Loading dataset...")
    dm = PolicyGuideDataModule(datasets=cfg.datasets, training_repo_root=zarr_path,
                               root_data_dir=zarr_path, transforms=cfg.transforms)
    dm.setup()
    dataset = dm.train_dataloader()

    # Load annotations
    print("Loading annotations...")
    annotations = np.load(zarr_path + "/training/lang_annotations/auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]['language']['ann']
    # print(f"Total samples in dataset: {len(dataset)}")

    static_win = "Static Camera"
    gripper_win = "Gripper Camera"

    for idx, batch in enumerate(dataset):
        if idx == 0:
            print(f"Batch Keys: {batch.keys()}")
        if batch["action"].shape[0] != cfg.datasets.batch_size:
            print(f"⚠️ Batch {idx} has {batch['action'].shape[0]} samples, expected {cfg.datasets.batch_size}.")
            
        # --- ANALYSIS ---
        if analyze:
            analyze_and_save(batch["action"].numpy(), "Action", idx)
            analyze_and_save(batch["observation.state"].numpy(), "State", idx)

            for cam_key in ["observation.image_static", "observation.image_wrist"]:
                images = batch[cam_key].numpy()  # Shape: (B, VIEWS, H, W, C)
                B, V, C, H, W = images.shape
                for view in range(V):
                    for channel in range(C):
                        img_data = images[:, view, channel, :, :]  # shape: (B, H, W)
                        flat = img_data.flatten()
                        name = f"{cam_key.replace('.', '_')}_view{view}_ch{channel}"
                        analyze_and_save(flat, name, idx)

        for i in range(batch["observation.image_static"].shape[0]):
            img = batch["observation.image_static"][i, 0, ...]
            img2 = batch["observation.image_static"][i, 1, ...]
            img_gripper = batch["observation.image_wrist"][i, 1, ...]

            action = batch["action"]
            state = batch["observation.state"]
            ann_idx = int(batch["language"][0])
            language = annotations[ann_idx]

            gif_name = language

            if gui:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        print(f"{gif_name}_{idx}")
        if not gui:
            save_images_and_create_gif(images, gif_name=f"{gif_name}_{idx}.gif")
            images = []
            labels = []

    if gui:
        cv2.destroyAllWindows()


def nan_batch_analyzer():
    # Set the directory containing your .pkl files
    directory = '/home/choudhue/PolicyGuide/results/runs/2025-06-16/15-29-44'

    # Collect all .pkl files
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    # Load each .pkl file and store contents in a dictionary
    for filename in tqdm.tqdm(pkl_files):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            try:
                data = pickle.load(file)
                idx = re.search(r'batch_(\d+)', filename)
                analyze_and_save(data["observation.state"].cpu().numpy(), "State", idx)

                for cam_key in ["observation.image_static", "observation.image_wrist"]:
                    images = data[cam_key].cpu().numpy()  # Shape: (B, VIEWS, H, W, C)
                    B, V, C, H, W = images.shape
                    for view in range(V):
                        for channel in range(C):
                            img_data = images[:, view, channel, :, :]  # shape: (B, H, W)
                            flat = img_data.flatten()
                            name = f"{cam_key.replace('.', '_')}_view{view}_ch{channel}"
                            analyze_and_save(flat, name, idx)
            except Exception as e:
                print(f"Error loading {filename}: {e}")



def analyze_and_save(array, name: str, batch_idx: int, save_dir: str = "/home/choudhue/PolicyGuide/viz/nan_histograms"):
    """Analyzes a tensor for anomalies and saves a histogram plot."""
    os.makedirs(save_dir, exist_ok=True)
    anomaly = False
    # print(f"\n[{name} Stats] (Batch {batch_idx})")

    # Check if the input is a file path
    if isinstance(array, str) and array.endswith('.pkl'):
        with open(array, 'rb') as f:
            array = pickle.load(f)
  
    lovely(array)

    #check datatype for float16
    if isinstance(array, np.ndarray) and array.dtype != np.float16:
        print(f"❗ {name} is not float16, dtype: {array.dtype}")

    # Anomaly Report
    # print(f"--- {name} Anomaly Report (Batch {batch_idx}) ---")
    if np.isnan(array).any():
        print("❌ Contains NaNs")
        anomaly = True
    if np.isinf(array).any():
        print("❌ Contains Infs")
        anomaly = True
    zero_count = np.sum(array == 0)
    if zero_count > 0:
        print(f"⚠️ Contains {zero_count} zeros")
        anomaly = True
    
    # Anomaly heuristics
    std = np.std(array)
    mean = np.mean(array)

    if std < 1e-3:
        print("⚠️ Very low std: likely constant or dead sensor")
        anomaly = True
    # if std > 5 * mean and mean > 0.01:
    #     print("❗ High std relative to mean: possible outliers")
    #     anomaly = True
    if std > 5.0:
        print("❗ Extremely high std: scale or clipping issue")
        anomaly = True
    # print(f"Min: {np.min(array):.4f}, Max: {np.max(array):.4f}, Mean: {np.mean(array):.4f}, Std: {np.std(array):.4f}\n")
    
    if anomaly:
        # Histogram
        flat = array.flatten()
        plt.figure(figsize=(6, 4))
        plt.hist(flat, bins=50, alpha=0.75, color='steelblue')
        plt.title(f"{name} Distribution (Batch {batch_idx})")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{name}_hist_batch{batch_idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Histogram saved to {plot_path}")

def save_images_and_create_gif(images: List[np.ndarray], save_dir: str = "/home/choudhue/PolicyGuide/viz", gif_name: str = "rollout.gif", fps: int = 30, lang=None):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for idx, img in enumerate(images):
        img_path = save_path / f"frame_{idx:04d}_{lang[idx] if lang else ''}.png"
        imageio.imwrite(img_path, img)
        image_paths.append(img_path)

    gif_path = save_path / gif_name
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for img_path in image_paths:
            writer.append_data(imageio.imread(img_path))

    for img_path in image_paths:
        os.remove(img_path)

    print(f"GIF saved at {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", help="Visualize using GUI, otherwise save visualization as gif.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--analyze", help="Analyze tensors for NaNs, zeros, and outliers", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # visualize_dataset() #gui=args.gui, analyze=args.analyze
    nan_batch_analyzer()
    

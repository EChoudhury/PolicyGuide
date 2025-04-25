import os
from pathlib import Path
from typing import List
import cv2
import imageio
import numpy as np
from itpg.datasets.policy_guide_dataset import PolicyGuideDataset
from itpg.datasets.policy_guide_data_module import PolicyGuideDataModule
from omegaconf import OmegaConf, DictConfig
import hydra

@hydra.main(config_path="/home/choudhue/PolicyGuide/conf/datamodule", config_name="default")
def visualize_dataset(cfg: DictConfig) -> None:
    # Path to the Zarr dataset
    zarr_path = "/home/choudhue/PolicyGuide/dataset/calvin_D_1T_dataset"

    obs_space = {
        "rgb_obs": ['rgb_static'],
        "depth_obs": [],
        "state_obs": ['robot_obs'],
        "actions": ['actions'],
        "language": ['language']
    }

    # Initialize the PolicyGuideDataset
    print("Loading dataset...")
    # print(f"Config: {OmegaConf.to_yaml(cfg)}")
    dm = PolicyGuideDataModule(datasets=cfg.datasets, training_repo_root=zarr_path, root_data_dir=zarr_path, transforms=cfg.transforms)
    # print(cfg.datasets)
    dm.setup()  # or just `setup()` if you're not specifying stage

    dataset = dm.train_dataloader()

    print("Loading annotations...")
    # load annotations from training folder
    annotations = np.load(zarr_path + "/training/lang_annotations/auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]['language']['ann']

    # Print the total number of samples in the dataset
    print(f"Total samples in dataset: {len(dataset)}")

    # Create a single OpenCV window
    window_name = "Dataset Visualization"
    cv2.namedWindow(window_name)
    # images = []

    # Iterate through the dataset and visualize the images
    for idx, batch in enumerate(dataset):
        if idx == 0:
            print(f"Batch Keys: {batch.keys()}")
        for i in range(batch["observation.image_static"].shape[0]):
            # Extract the static image observation
            img = batch["observation.image_static"][i,1, ...] # Shape: (height, width, channels)
            # img = img.permute(1, 2, 0) # Convert from (C, H, W) to (H, W, C)
            print(img.shape)
            # Denormalize the image (reverse the normalization process)
            # mean = [0.5]  # Replace with the actual mean used during normalization
            # std = [0.5]   # Replace with the actual std used during normalization
            # for c in range(img.shape[0]):  # Iterate over channels
            #     img[c, :, :] = img[c, :, :] * std[0] + mean[0]

            # Convert the image to a format suitable for OpenCV
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = (img * 255).clamp(0, 255).byte().cpu().numpy().astype(np.uint8) # Scale to [0, 255] and convert to uint8
            # img = img.numpy()  # Convert from PyTorch tensor to NumPy array

            action = batch["action"]  # Shape: (8,)
            state = batch["observation.state"] # Shape: (8,)
            ann_idx = int(batch["language"][0])
            language = annotations[ann_idx]  # Shape: (sequence_length,)
            # Convert the image to a format suitable for OpenCV (if needed)
            # if isinstance(img, np.ndarray):
            #     img = img.astype(np.uint8)  # Ensure the image is in uint8 format
            # elif hasattr(img, "numpy"):  # If it's a PyTorch tensor
            #     img = img.numpy().astype(np.uint8)

            # Display the image in the same OpenCV window
            cv2.imshow(window_name, img)
            # images.append(img)
            print(f"Sample {idx}: \nAction: {action}, \nState: {state}, \nLanguage: {language}\n\n")

            # Wait for a short duration or until 'q' is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break
         # save images for gif
        # save_images_and_create_gif(images)
    # Destroy the OpenCV window after visualization
    cv2.destroyAllWindows()

def save_images_and_create_gif(images: List[np.ndarray], save_dir: str = "/home/choudhue/PolicyGuide/viz", gif_name: str = "rollout.gif", fps: int = 30):
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
            writer.append_data(imageio.imread(img_path))

    print(f"GIF saved at {gif_path}")

if __name__ == "__main__":
    visualize_dataset()

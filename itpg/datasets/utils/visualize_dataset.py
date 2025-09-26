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

@hydra.main(config_path="/home/choudhue/PolicyGuide/conf/datamodule", config_name="default")
def visualize_dataset(cfg: DictConfig, gui=False) -> None:
    # Path to the Zarr dataset
    zarr_path = "/home/choudhue/PolicyGuide/dataset/calvin_D_play_dataset"

    # Initialize the PolicyGuideDataset
    print("Loading dataset...")
    # print(f"Config: {OmegaConf.to_yaml(cfg)}")
    dm = PolicyGuideDataModule(datasets=cfg.datasets, training_repo_root=zarr_path, root_data_dir=zarr_path, transforms=cfg.transforms)
    # print(cfg.datasets)
    dm.setup()  # or just `setup()` if you're not specifying stage

    dataset = dm.train_dataloader()

    print("Loading annotations...")
    # load annotations from training folder
    # annotations = np.load(zarr_path + "/training/lang_annotations/auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]['language']['ann']

    # Print the total number of samples in the dataset
    # print(f"Total samples in dataset: {len(dataset)}")

    # Create a single OpenCV window
    static_win = "Static Camera"
    gripper_win = "Gripper Camera"
    if gui:
        cv2.namedWindow(static_win)
        cv2.namedWindow(gripper_win)
    else:
        images = []
        labels = []
        gif_name = ""

    # Iterate through the dataset and visualize the images
    for idx, batch in enumerate(dataset):
        # start_time = time.time()
        if idx == 0:
            print(f"Batch Keys: {batch.keys()}")
        for i in range(batch["observation.image_static"].shape[0]):
            # Extract the static image observation
            img = batch["observation.image_static"][i,0, ...] # Shape: (height, width, channels)
            img2 = batch["observation.image_static"][i,1, ...] 
            img_gripper = batch["observation.image_wrist"][i,1, ...] # Shape: (height, width, channels)

            # # Convert the image to a format suitable for OpenCV
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = (img * 255).clamp(0, 255).byte().cpu().numpy().astype(np.uint8) # Scale to [0, 255] and convert to uint8

            # img2 = img2.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            # img2 = (img2 * 255).clamp(0, 255).byte().cpu().numpy().astype(np.uint8) # Scale to [0, 255] and convert to uint8

            # # Convert the image to a format suitable for OpenCV
            img_gripper = img_gripper.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img_gripper = (img_gripper * 255).clamp(0, 255).byte().cpu().numpy().astype(np.uint8) # Scale to [0, 255] and convert to uint8

            action = batch["action"]  # Shape: (8,)
            action_one = batch["action"][0]
            state = batch["observation.state"] # Shape: (8,)
            # ann_idx = int(batch["language"][0])
            # language = annotations[ann_idx] # Shape: (sequence_length,)

            # Display the image in the same OpenCV window
            if gui:
                cv2.imshow(static_win, img)
                cv2.imshow(gripper_win, img_gripper)
            else:
                images.append(img)
                # labels.append(language + "_1")
                images.append(img_gripper)
                # labels.append(language + "_2")
            # Raw Data  
            # print(f"Sample {idx}: \nAction: {action}, \nState: {state}, \nLanguage: {language}\n\n")
            gif_name = f"{idx}_{i}"
            # gif_name = language
            
            # Wait for a short duration or until 'q' is pressed
            if gui:
                if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                    break
        # save images for gif
        print(f"End of batch_{idx}")
        if not gui:
            save_images_and_create_gif(images, gif_name=f"{gif_name}_{idx}.gif")
            images = []
            labels = []

        # end_time = time.time()  # End the timer for this iteration
        # iteration_time = end_time - start_time  # Calculate the duration
        # print(f"Iteration {idx} : {iteration_time:.2f} seconds")
    # Destroy the OpenCV window after visualization
    if gui:
        cv2.destroyAllWindows()

def save_images_and_create_gif(images: List[np.ndarray], save_dir: str = "/home/choudhue/PolicyGuide/viz/example_rollout_play", gif_name: str = "rollout.gif", fps: int = 5, lang=None):
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
        if lang is None:
            img_path = save_path / f"frame_{idx:04d}.png"
        else:
            img_path = save_path / f"frame_{idx:04d}_{lang[idx]}.png"
        imageio.imwrite(img_path, img)
        image_paths.append(img_path)

    # Create GIF
    gif_path = save_path / gif_name
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for img_path in image_paths:
            writer.append_data(imageio.imread(img_path))

    # for img_path in image_paths:
    #     os.remove(img_path)

    print(f"GIF saved at {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", help="Visualize using gui, otherwise save visualization as gif.", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    visualize_dataset()

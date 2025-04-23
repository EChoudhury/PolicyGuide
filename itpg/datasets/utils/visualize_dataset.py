import cv2
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

    # Iterate through the dataset and visualize the images
    for idx, batch in enumerate(dataset):
        if idx == 0:
            print(f"Batch Keys: {batch.keys()}")
        for i in range(batch["observation.image_static"].shape[0]):
            # Extract the static image observation
            img = batch["observation.image_static"][i,1, ...] # Shape: (height, width, channels)
            # img = img.permute(1, 2, 0) # Convert from (C, H, W) to (H, W, C)
            print(img)
            # Denormalize the image (reverse the normalization process)
            mean = [0.5]  # Replace with the actual mean used during normalization
            std = [0.5]   # Replace with the actual std used during normalization
            for c in range(img.shape[0]):  # Iterate over channels
                img[c, :, :] = img[c, :, :] * std[0] + mean[0]

            # Convert the image to a format suitable for OpenCV
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = (img * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to uint8
            img = img.numpy()  # Convert from PyTorch tensor to NumPy array

            action = batch["action"]  # Shape: (8,)
            state = batch["observation.state"] # Shape: (8,)
            ann_idx = int(batch["language"][0])
            language = annotations[ann_idx]  # Shape: (sequence_length,)
            # Convert the image to a format suitable for OpenCV (if needed)
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)  # Ensure the image is in uint8 format
            elif hasattr(img, "numpy"):  # If it's a PyTorch tensor
                img = img.numpy().astype(np.uint8)

            # Display the image in the same OpenCV window
            cv2.imshow(window_name, img)
            print(f"Sample {idx}: \nAction: {action}, \nState: {state}, \nLanguage: {language}\n\n")

            # Wait for a short duration or until 'q' is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break

    # Destroy the OpenCV window after visualization
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_dataset()

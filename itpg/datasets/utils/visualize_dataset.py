import cv2
import numpy as np
from itpg.datasets.policy_guide_dataset import PolicyGuideDataset

# Path to the Zarr dataset
zarr_path = "/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training"

obs_space = {
    "rgb_obs": ['rgb_static'],
    "depth_obs": [],
    "state_obs": ['robot_obs'],
    "actions": ['actions'],
    "language": ['language']
}

prop_space = {
    "n_state_obs": 8,
    "keep_indices": [[0, 7], [14,15]],
    "robot_orientation_idx": [3, 6],
    "normalize": False,
    "normalize_robot_orientation": False,
}

# Initialize the PolicyGuideDataset
print("Loading dataset...")
dataset = PolicyGuideDataset(dataset_dir=zarr_path, obs_space=obs_space, proprio_state=prop_space, abs_datasets_dir=zarr_path, lang_folder="lang_annotations")

print("Loading annotations...")
# load annotations from training folder
annotations = np.load(zarr_path + "/lang_annotations/auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]['language']['ann']

# Print the total number of samples in the dataset
print(f"Total samples in dataset: {len(dataset)}")

# Create a single OpenCV window
window_name = "Dataset Visualization"
cv2.namedWindow(window_name)

# Iterate through the dataset and visualize the images
for idx, batch in enumerate(dataset):
    # Extract the static image observation
    img = batch["observation.image_static"][1, ...]  # Shape: (height, width, channels)
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


#Action from Evalutation: [[-0.0330, -0.1913,  0.4859,  1.0000,  0.0282,  0.5888,  0.9260]]
# Action: tensor([[[ 0.1308, -0.1967,  0.5056, -1.0000,  0.1895,  0.1931,  0.8631],
#          [ 0.1591, -0.1258,  0.4774, -1.0000,  0.1535,  0.2230,  0.8808],
#          [ 0.1529, -0.1555,  0.5479,  1.0000,  0.0699,  0.3577,  0.8330],
#          [ 0.1431, -0.1024,  0.4923,  1.0000, -0.0164,  0.3930,  0.7626],
#          [ 0.1995, -0.1597,  0.5087, -1.0000,  0.1246,  0.4062,  0.9116],
#          [ 0.0718, -0.1605,  0.4476, -1.0000,  0.2348,  0.4989,  0.8498],
#          [ 0.1464, -0.0738,  0.4934, -1.0000,  0.0967,  0.4763,  0.8072],
#          [ 0.1110, -0.1961,  0.4960, -1.0000,  0.1460,  0.3856,  0.8172]]],

# Sample 65: 
# State: tensor([[ 0.1743, -0.2330,  0.4011,  3.0955,  0.0632,  1.4823,  0.0800,  1.0000],
#         [ 0.1744, -0.2323,  0.3974,  3.0978,  0.0631,  1.4814,  0.0800,  1.0000]]), 
# Action: tensor([[ 0.1744, -0.2323,  0.3974,  3.0978,  0.0631,  1.4814,  1.0000],
#         [ 0.1749, -0.2309,  0.3924,  3.1004,  0.0611,  1.4762,  1.0000],
#         [ 0.1756, -0.2292,  0.3865,  3.1014,  0.0589,  1.4713,  1.0000],
#         [ 0.1760, -0.2278,  0.3817,  3.0999,  0.0585,  1.4705,  1.0000],
#         [ 0.1762, -0.2262,  0.3774,  3.0963,  0.0559,  1.4710,  1.0000],
#         [ 0.1764, -0.2240,  0.3725,  3.0916,  0.0507,  1.4727,  1.0000],
#         [ 0.1761, -0.2224,  0.3687,  3.0859,  0.0476,  1.4773, -1.0000],
#         [ 0.1756, -0.2202,  0.3637,  3.0751,  0.0433,  1.4838, -1.0000],
#         [ 0.1752, -0.2177,  0.3586,  3.0616,  0.0375,  1.4865, -1.0000],
#         [ 0.1745, -0.2153,  0.3543,  3.0506,  0.0319,  1.4890, -1.0000],
#         [ 0.1736, -0.2130,  0.3505,  3.0384,  0.0260,  1.4946, -1.0000],
#         [ 0.1726, -0.2111,  0.3474,  3.0257,  0.0212,  1.4997, -1.0000],
#         [ 0.1716, -0.2104,  0.3446,  3.0185,  0.0194,  1.5082, -1.0000],
#         [ 0.1708, -0.2099,  0.3426,  3.0103,  0.0151,  1.5152, -1.0000],
#         [ 0.1702, -0.2101,  0.3402,  3.0060,  0.0110,  1.5138, -1.0000],
#         [ 0.1696, -0.2102,  0.3383,  3.0063,  0.0083,  1.5117, -1.0000]]), 

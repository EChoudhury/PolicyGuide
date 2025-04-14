import cv2
import zarr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from itpg.datasets.policy_guide_dataset import PolicyGuideDataset

# Load the Zarr dataset
zarr_path = "/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training"

dataset = PolicyGuideDataset(
        datasets_dir="/home/choudhue/PolicyGuide/dataset/calvin_D_3T_dataset/training"
    )

# Assume the dataset has 'images' and 'annotations' arrays
 # shape: (num_images, height, width, channels)
# annotations = dataset["annotations"]  # shape: (num_images,)
print(len(dataset))
for batch in dataset:
    img = batch["observation"]['image_static'] 
    # cv2.imshow('Image Sequence', img.numpy())
    # if cv2.waitKey(int(0.5 * 1000)) & 0xFF == ord('q'):
    #     break

# cv2.destroyAllWindows()
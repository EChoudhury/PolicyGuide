import imageio
import os
from pathlib import Path
from typing import List
import numpy as np

def save_images_and_create_gif(images: List[np.ndarray], save_dir: str, gif_name: str = "rollout.gif", fps: int = 10):
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
            writer.append_data(imageio.v2.imread(img_path))
    
    for img_path in image_paths:
        os.remove(img_path)

    print(f"GIF saved at {gif_path}")


def combine_gifs(folder_path: str, output_gif_name: str = "combined_sequences.gif", fps: int = 30):
    """
    Combine multiple GIFs into a single GIF.

    Args:
        folder_path (str): Path to the folder containing the GIFs.
        output_gif_name (str): Name of the output combined GIF file.
        fps (int): Frames per second for the combined GIF.
    """
    gif_files = sorted(Path(folder_path).glob("*.gif"))
    images = []

    for gif_file in gif_files:
        with imageio.get_reader(gif_file) as reader:
            for frame in reader:
                images.append(frame)

    output_gif_path = Path(folder_path) / output_gif_name
    with imageio.get_writer(output_gif_path, mode="I", fps=fps) as writer:
        for img in images:
            writer.append_data(img)

    print(f"Combined GIF saved at {output_gif_path}")

if __name__ == "__main__":
    # Example usage
    save_dir = "/home/choudhue/PolicyGuide/results/runs/2025-04-29/16-08-48/eval_viz_20250509_153616"

    # Generate some random images for demonstration
    # images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]  # Replace with actual images
    # save_images_and_create_gif(images, save_dir)

    # Combine GIFs from the same directory
    combine_gifs(save_dir)
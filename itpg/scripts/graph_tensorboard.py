import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def inspect_tensorboard_files(directory):
    """
    Inspect TensorBoard event files in the given directory and print the available data.

    Args:
        directory (str): Path to the directory containing TensorBoard event files.
    """
    # Iterate through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if "events.out.tfevents" in file:  # Check if the file is a TensorBoard event file
                file_path = os.path.join(root, file)
                print(f"\nInspecting file: {file_path}")
                
                # Load the event file
                event_acc = event_accumulator.EventAccumulator(file_path)
                event_acc.Reload()

                # Get all available tags
                tags = event_acc.Tags()
                print(f"Available tags: {tags}")

                # Inspect scalar data
                if 'scalars' in tags:
                    print("\nScalar data:")
                    for tag in tags['scalars']:
                        print(f"  Tag: {tag}")
                        scalars = event_acc.Scalars(tag)
                        print(f"    Number of entries: {len(scalars)}")
                        if scalars:
                            print(f"    First entry: Step={scalars[0].step}, Value={scalars[0].value}")
                            print(f"    Last entry: Step={scalars[-1].step}, Value={scalars[-1].value}")

                # Inspect other types of data (e.g., histograms, images, etc.)
                for data_type in ['histograms', 'images', 'audio', 'tensors']:
                    if data_type in tags and tags[data_type]:
                        print(f"\n{data_type.capitalize()} data:")
                        for tag in tags[data_type]:
                            print(f"  Tag: {tag}")

def plot_loss_from_tensorboard(directory):
    """
    Plot train/loss and valid/loss from TensorBoard event files using Matplotlib.

    Args:
        directory (str): Path to the directory containing TensorBoard event files.
    """
    train_loss = []
    valid_loss = []
    steps = []

    # Iterate through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if "events.out.tfevents" in file:  # Check if the file is a TensorBoard event file
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Load the event file
                event_acc = event_accumulator.EventAccumulator(file_path)
                event_acc.Reload()

                # Extract train/loss and valid/loss
                tags = event_acc.Tags()
                if 'scalars' in tags:
                    if 'train/loss' in tags['scalars']:
                        train_loss = event_acc.Scalars('train/loss')
                    if 'valid/loss' in tags['scalars']:
                        valid_loss = event_acc.Scalars('valid/loss')

    # Extract steps and values
    if train_loss:
        train_steps = [scalar.step for scalar in train_loss]
        train_values = [scalar.value for scalar in train_loss]
    else:
        train_steps, train_values = [], []

    if valid_loss:
        valid_steps = [scalar.step for scalar in valid_loss]
        valid_values = [scalar.value for scalar in valid_loss]
    else:
        valid_steps, valid_values = [], []

    # Plot the losses
    plt.figure(figsize=(10, 6))
    if train_steps and train_values:
        plt.plot(train_steps, train_values, label='Train Loss', color='blue')
    if valid_steps and valid_values:
        plt.plot(valid_steps, valid_values, label='Validation Loss', color='orange')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_valid_loss.png'))
    # plt.show()
    print("Plot saved as 'train_valid_loss.png'")

if __name__ == "__main__":
    # Specify the directory containing TensorBoard event files
    tensorboard_dir = "/home/choudhue/PolicyGuide/results/runs/2025-04-29/16-08-48/tensorboard_data"
    
    # Inspect TensorBoard files
    # inspect_tensorboard_files(tensorboard_dir)
    
    # Plot train/loss and valid/loss
    plot_loss_from_tensorboard(tensorboard_dir)
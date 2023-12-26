import matplotlib.pyplot as plt
from torchvision import transforms
import torch


# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.show()

# function to collate data
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    # Replace label strings with label numeric values
    target_map = {'valid': 0, 'above': 1, 'parallel': 2}
    target = [target_map[label] for label in target]

    return torch.stack(data), torch.tensor(target)

# Function to plot losses
def plot_losses(train_losses, val_losses):
    """
    Plot the training and validation losses.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torchvision.transforms.functional import resize
from torchvision.io import ImageReadMode
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import pandas as pd
import os

# Create a custom image dataset using the existing csv file
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.base_dir = base_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_labels.iloc[idx, 2])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "/home/samuele/Documenti/GitHub/Back_Squats_IPF/dataset"
    # Path to csv dataframe file
    df_file = "/home/samuele/Documenti/GitHub/Back_Squats_IPF/image_labels.csv"

    # Let's check if the code from above works with a dataloader!
    
    # stack expects each tensor to be equal size so we resize the images
    def resize_transform(img):
        return resize(img, size=(224,224))

    dataset = CustomImageDataset(df_file, dataset_path, transform=resize_transform)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}") # train_labels is a tuple
    print(type(train_features))
    img = train_features[0]
    label = train_labels[0]

    # use permute to flip the channels so that we have a shape (400,400,3). Before it was (3,400,400)
    plt.imshow(img.permute(1, 2, 0))

    plt.show()
    print(f"Label: {label}")

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
    
# Tweaking the function to plot losses

def plot_losses(train_losses, val_losses, train_accuracy=None, val_accuracy=None):
    """
    Plot the training and validation losses and optionally accuracies.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracy (list, optional): List of training accuracies.
        val_accuracy (list, optional): List of validation accuracies.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies if provided
    if train_accuracy is not None and val_accuracy is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()


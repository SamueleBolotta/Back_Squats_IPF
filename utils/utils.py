import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torchvision.transforms.functional import resize
from torchvision.io import ImageReadMode
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

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

def image_dataframe_from_path(image_dir):
    """
    Walks a directory to get all the images in it and returns a dataframe
    with columns 'name' and 'image_path'
    
    Args:
        image_dir: image directory
    """
    image_list = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                if not "Screenshot" in file: # make sure that the data we are using is labeled with a name
                    image_path = os.path.join(root, file)
                    name, _ = os.path.splitext(file)
                    image_list.append({'name': name, 'image_path': image_path, 'image_root':root})

    # Convert the list to a pandas dataframe
    return pd.DataFrame(image_list)


def save_images(target_df, dataset_name):
    """
    Saves images to a target directory specified in the given target_df (dataframe)
    
    Args:
        target_df: name of the dataframe
        dataset_name: string
    """
    total_num_images = len(target_df)
    pbar = tqdm(total=total_num_images)
    for i, img in target_df.iterrows():
        with Image.open(img["image_path"]) as image:
            if image.mode != 'RGB':
                image.convert('RGB')
            new_path = img["image_root"].replace('Original', dataset_name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            img_dir = os.path.join(new_path, f'{img["name"]}.png')
            image.save(img_dir)
        pbar.update(1)
    pbar.close()

def build_rgb_dataset(image_dir, train_ratio):
    """
    Splits original dataset from image_dir into train and test datasets
    Converts all the images from the new datasets to RGB format and saves them

    Args:
        image_dir: original dataset directory
        train_ratio: ratio of training data (test dataset ratio is calculated as 1-train_ratio)
    """
    # Split the dataframe into training and test sets
    train, test = train_test_split(image_dataframe_from_path(image_dir), 
                                   train_size=train_ratio, 
                                   test_size=1-train_ratio, 
                                   shuffle=True)
    
    return train, test

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
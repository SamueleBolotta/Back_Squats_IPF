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
    dataset_path = "/home/juancm/trento/SIV/siv_project/dataset_backup_13-12-2023/Powerlifting_Dataset"
    # Path to csv dataframe file
    df_file = "/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/image_labels.csv"

    # Let's check if the code from above works with a dataloader!
    
    # we'll use a simple transformation
    def resize_transform(img):
        return resize(img, size=(400,400))

    dataset = CustomImageDataset(df_file, dataset_path, transform=resize_transform)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}") # train_labels is a tuple
    
    img = train_features[0]
    label = train_labels[0]

    # use permute to flip the channels so that we have a shape (400,400,3). Before it was (3,400,400)
    plt.imshow(img.permute(1, 2, 0))

    plt.show()
    print(f"Label: {label}")
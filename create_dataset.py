#%%
import os
import pandas as pd

#%%
# Path to your dataset
dataset_path = "/home/juancm/trento/SIV/siv_project/dataset_backup_13-12-2023/Powerlifting_Dataset"

image_labels = []

# Traverse the directory structure
for root, dirs, files in os.walk(dataset_path):
    # The subfolder name is the label
    label = os.path.basename(root)
    
    # for each file in the subfolder
    for file in files:
        # Only consider image files
        if file.endswith('.jpg') or file.endswith('.png'):
            if not file.startswith('Screenshot'):
                # Get the paths from root
                path_from_root = os.path.relpath(os.path.join(root, file), dataset_path).replace(" ","\ ")
                # Add the file and label to the list
                image_labels.append((file, label, path_from_root))

# Convert the list to a DataFrame
df = pd.DataFrame(image_labels, columns=["image", "label", "relative path"])

# Save the DataFrame as a CSV file
df
# %%
new_labels = {'Frontal_Above_parallel':'above', 'Lateral_Above_parallel':'above',
              'Frontal_Parallel':'parallel', 'Lateral_Parallel':'parallel',
              'Frontal_Valid':'valid', 'Lateral_Valid':'valid'}

df['label'] = df['label'].replace(new_labels)
df.to_csv("image_labels.csv", index=False)
df
# %%
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset

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

# %%
from torchvision.transforms.functional import resize
# Path to csv dataframe file
df_file = "/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/image_labels.csv"

def resize_transform(img):
    return resize(img, size=(400,400))

dataset = CustomImageDataset(df_file, dataset_path, transform=resize_transform)
# %%
image, label = dataset[0]
print(type(image))
print(label)
print(image.shape)
# %%

from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# %%
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]
label = train_labels[0]

plt.imshow(img.permute(1, 2, 0))

plt.show()
print(f"Label: {label}")
# %%
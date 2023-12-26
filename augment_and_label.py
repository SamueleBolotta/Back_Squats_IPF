import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from torchvision import datasets
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.2, contrast=0.5),  # Randomly change the brightness and contrast
    # transforms.ToTensor(),
])

image_labels = []
# get the current working directory
current_dir = os.path.abspath(os.getcwd())

for sex in ("Men", "Women"):
    # Path to your original dataset
    original_dataset_path = f'{current_dir}/dataset/{sex}/Original'
    # Path to save augmented images
    augmented_dataset_path = f'{current_dir}/dataset/{sex}/Augmented'

    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path)

    # Traverse the directory structure
    ## Pay attention to the fact that the images are in format RGBA, which is not supported by JPEG
    ## Convert it to RGB before saving

    for root, dirs, files in os.walk(original_dataset_path):
        # The subfolder name is the label
        label = os.path.basename(root)

        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # avoid using images that are unlabelled
                if not "Screenshot" in file:
                    file_path = os.path.join(root, file)
                    image = Image.open(file_path)
                    
                    # Convert image to RGB if it has more than 3 channels
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Apply transformations and save augmented images
                    path_with_label = os.path.join(augmented_dataset_path, label)
                    for i in range(5):
                        if not os.path.exists(path_with_label):
                            os.makedirs(path_with_label)
                        transformed_image = transform(image)
                        # transformed_image = transforms.ToPILImage()(transformed_image)
                        transformed_image.save(os.path.join(path_with_label, f'{i}_{file}'))
    # Create CSV file
    # Path to your dataset (including augmented images)
    dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/dataset'

    # Traverse the directory structure
    for root, dirs, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # avoid using images that are unlabelled
                if not "Screenshot" in file:
                    # Get the paths from root
                    path_from_root = os.path.relpath(os.path.join(root, file), dataset_path)
                    # print(root)
                    # print(path_from_root)
                    # Add the file and label to the list
                    image_labels.append((file, label, path_from_root))

df = pd.DataFrame(image_labels, columns=["image", "label", "relative path"])

new_labels = {'Frontal_Above_parallel':'above', 'Lateral_Above_parallel':'above',
              'Frontal_Parallel':'parallel', 'Lateral_Parallel':'parallel',
              'Frontal_Valid':'valid', 'Lateral_Valid':'valid',
              'Frontal - Above parallel':'above', 'Lateral - Above parallel':'above',
              'Frontal - Parallel':'parallel', 'Lateral - Parallel':'parallel',
              'Frontal - Valid':'valid', 'Lateral - Valid':'valid'}

# Save the DataFrame to a CSV file
df['label'] = df['label'].replace(new_labels)
df.to_csv('image_labels.csv', index=False)

# Create a dataset using the ImageFolder class
dataset = datasets.ImageFolder(root=f'{current_dir}/dataset', transform=transform)

# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    #img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {dataset.classes[label]}')
    plt.show()

# Visualize the transformations for the first ten images in the dataset
# for i in range(10):
#     visualize_transformations(dataset, i)
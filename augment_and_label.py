import os
from PIL import Image
from torchvision import transforms
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly change the brightness and contrast
])

# Path to your original dataset
original_dataset_path = 'C:\Users\bolot\OneDrive\Desktop\Back_Squats_IPF\Men\Original'
# Path to save augmented images
augmented_dataset_path = 'C:\Users\bolot\OneDrive\Desktop\Back_Squats_IPF\Men\Augmented'

if not os.path.exists(augmented_dataset_path):
    os.makedirs(augmented_dataset_path)

# Traverse the directory structure
for root, dirs, files in os.walk(original_dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            # Apply transformations and save augmented images
            for i in range(5):  # Number of augmented versions to create
                transformed_image = transform(image)
                new_file_path = os.path.join(augmented_dataset_path, f'aug_{i}_{file}')
                transformed_image.save(new_file_path)

# Create CSV file
import pandas as pd

# Path to your dataset (including augmented images)
dataset_path = 'C:\Users\bolot\OneDrive\Desktop\Back_Squats_IPF\Men'

# Continue with the rest of your code
image_labels = []

# Traverse the directory structure
for root, dirs, files in os.walk(dataset_path):
    label = os.path.basename(root)
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(root, file)
            image_labels.append((file_path, label))

df = pd.DataFrame(image_labels, columns=["image", "label"])

# Save the DataFrame to a CSV file
df.to_csv('image_labels.csv', index=False)

# Create a dataset using the ImageFolder class
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {dataset.classes[label]}')
    plt.show()

# Visualize the transformations for the first image in the dataset
visualize_transformations(dataset, 0)




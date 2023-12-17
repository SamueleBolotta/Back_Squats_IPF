import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import datasets
from torchvision import transforms


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly change the brightness and contrast
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
])

image_labels = []
# get the current working directory
current_dir = os.path.abspath(os.getcwd())

for gender in ("Men", "Women"):
    # Path to your original dataset
    original_dataset_path = f'{current_dir}/dataset/{gender}/Original'
    # Path to save augmented images
    augmented_dataset_path = f'{current_dir}/dataset/{gender}/Augmented'

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
                        # Convert the PyTorch tensor back to PIL Image
                        transformed_image_pil = transforms.ToPILImage()(transformed_image)
                        transformed_image_pil.save(os.path.join(augmented_dataset_path, f'{file.split(".")[0]}_transformed.jpg'))    # Create CSV file
                        
    # Path to your dataset (including augmented images)
    dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/dataset/{}'.format(gender)

    # Traverse the directory structure
    for root, dirs, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # avoid using images that are unlabelled
                if not "Screenshot" in file:
                    # Get the paths from root
                    path_from_root = os.path.relpath(os.path.join(root, file), dataset_path)
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

print(df)

# Create a dataset using the ImageFolder class
dataset = datasets.ImageFolder(root=f'{current_dir}/dataset', transform=transform)

# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    #img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {dataset.classes[label]}')
    plt.show()



# Now we set up the training and validation data loaders
from torch.utils.data import DataLoader, random_split

# Define the batch size
batch_size = 32

# Set the random seed
random_seed = 42

# Set the validation split
validation_split = .2

# Calculate the validation split based on the number of images in the dataset
dataset_size = len(dataset)

# Assuming you have a dataset class, adjust the imports and dataset initialization accordingly

# Calculate the split based on the length of the dataset
val_size = int(np.floor(validation_split * dataset_size))

# Calculate the training split based on the length of the dataset
train_size = dataset_size - val_size

# Create the training and validation datasets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the training and validation data loaders making sure that the data is tensors and not PIl images

# Create the training and validation data loaders
# Start the training loop
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    
    # Convert PIL images to tensors
    data = [transforms.ToTensor()(img) for img in data]

    return torch.stack(data), torch.tensor(target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Now we set up the test data loader

# Assuming you have a test_dataset, adjust the imports and dataset initialization accordingly
# For example, if you have a directory structure similar to the training/validation data, you can use ImageFolder

# Path to your test dataset
test_dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/Women'

# Define the transformation to convert PIL images to tensors for the test set
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (adjust as needed)
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
])

# Create the test dataset
test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=test_transform)

# Create the test data loader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Now we set up the model

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze the parameters
for param in model.parameters():
    param.requires_grad = False

# Change the final layer of the model
#Three outputs: above parallel, parallel, below parallel
model.fc = nn.Linear(512, 3)

# Transfer the model to the GPU
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Define the number of epochs
n_epochs = 25

# Now we train the model

# Initialize the best validation accuracy
best_val_accuracy = 0


for epoch in range(n_epochs):
    # Set the model to training mode
    model.train()
    
    # Initialize the running loss and correct predictions
    running_loss = 0.0
    correct_predictions = 0.0
    
    # Loop over the training data loader
    for data, target in train_loader:
        # Transfer the data to the GPU
        data = data.to(device)
        target = target.to(device)
        
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        output = model(data)
        
        # Calculate the loss
        loss = criterion(output, target)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Calculate the predictions
        predictions = torch.argmax(output, dim=1)
        
        # Update the number of correct predictions
        correct_predictions += torch.sum(predictions == target).item()
        
        # Update the running loss
        running_loss += loss.item() * data.size(0)
        
    # Calculate the average training loss and accuracy
    train_loss = running_loss / len(train_dataset)
    train_accuracy = correct_predictions / len(train_dataset)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize the running loss and correct predictions
    running_loss = 0.0
    correct_predictions = 0.0
    
    # Loop over the validation data loader
    for data, target in val_loader:
        # Transfer the data to the GPU
        data = data.to(device)
        target = target.to(device)
        
        # Perform forward pass
        output = model(data)
        
        # Calculate the loss
        loss = criterion(output, target)
        
        # Calculate the predictions
        predictions = torch.argmax(output, dim=1)
        
        # Update the number of correct predictions
        correct_predictions += torch.sum(predictions == target).item()
        
        # Update the running loss
        running_loss += loss.item() * data.size(0)
        
    # Calculate the average validation loss and accuracy
    val_loss = running_loss / len(val_dataset)
    val_accuracy = correct_predictions / len(val_dataset)
    
    # Print the metrics for every epoch
    print(f'Epoch: {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | Val loss: {val_loss:.4f}')

# Now we test the model
# Set the model to evaluation mode
model.eval()
                                                                                                                           
# Initialize the running loss and correct predictions
running_loss = 0.0
correct_predictions = 0.0
                                                                                                                           
# Loop over the test data loader

for data, target in test_loader:
    # Transfer the data to the GPU
    data = data.to(device)
    target = target.to(device)
                                                                                                                           
    # Perform forward pass
    output = model(data)
                                                                                                                           
    # Calculate the loss
    loss = criterion(output, target)
                                                                                                                           
    # Calculate the predictions
    predictions = torch.argmax(output, dim=1)
                                                                                                                           
    # Update the number of correct predictions
    correct_predictions += torch.sum(predictions == target).item()
                                                                                                                           
    # Update the running loss
    running_loss += loss.item() * data.size(0)
                                                                                                                           
# Calculate the average test loss and accuracy
test_loss = running_loss / len(test_dataset)
test_accuracy = correct_predictions / len(test_dataset)
                                                                                                                           
# Print the test metrics
print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}')

# Now we save the model
# Save the model
torch.save(model.state_dict(), 'model.pt')

# Now we load the model
# Load the model
model.load_state_dict(torch.load('model.pt'))

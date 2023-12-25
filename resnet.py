#%%
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

current_dir = os.getcwd()

#%%
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly change the brightness and contrast
    # transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
])

#%%
# Create a dataset using the ImageFolder class
# dataset = datasets.ImageFolder(root=f'{current_dir}/dataset', transform=transform)
from create_custom_dataset import CustomImageDataset
# Path to your dataset
dataset_path = f'{current_dir}/dataset/'
# Path to csv dataframe file
df_file = f'{current_dir}/image_labels.csv'

dataset = CustomImageDataset(df_file, dataset_path, transform=transform)

# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.show()

visualize_transformations(dataset, 0)
#%%

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
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)
#%%
print(type(train_dataset))

#%%
# generator1 = torch.Generator().manual_seed(42)
# generator2 = torch.Generator().manual_seed(42)
# random_split(range(10), [3, 7], generator=generator1)
# random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

# Create the training and validation data loaders making sure that the data is tensors and not PIl images

# Create the training and validation data loaders
# Start the training loop
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    
    # Convert PIL images to tensors
    # data = [transforms.ToTensor()(img) for img in data]

    # Replace label strings with label numeric values
    target_map = {'valid': 0, 'above': 1, 'parallel': 2}
    target = [target_map[label] for label in target]

    return torch.stack(data), torch.tensor(target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
#%%
next(iter(train_loader))

#%%
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {len(train_labels)}") # train_labels is a tuple

img = train_features[0]
label = train_labels[0]

#%%
# use permute to flip the channels so that we have a shape (400,400,3). Before it was (3,400,400)
plt.imshow(img.permute(1, 2, 0))

plt.show()
print(f"Label: {label}")
#%%
# Now we set up the model

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
#%%
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
n_epochs = 75

# Now we train the model

# Initialize the best validation accuracy
best_val_accuracy = 0

#%%
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
        output = model(data.float())
        
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
        output = model(data.float())
        
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
    output = model(data.float())
                                                                                                                           
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

# # Now we save the model
# # Save the model
torch.save(model.state_dict(), 'model.pt')

# Now we load the model
# Load the model
model.load_state_dict(torch.load('model.pt'))

# %%

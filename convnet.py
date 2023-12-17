import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from torchvision import datasets
import matplotlib.pyplot as plt
import torch
import pytorchtools
import os
from PIL import Image
from torchvision import transforms
import numpy as np

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
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
])

# Path to your original dataset
original_dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/Men/Original'
# Path to save augmented images
augmented_dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/Men/Augmented'

if not os.path.exists(augmented_dataset_path):
    os.makedirs(augmented_dataset_path)

# Traverse the directory structure
for root, dirs, files in os.walk(original_dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            
            # Convert image to RGB if it has more than 3 channels
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            transformed_image = transform(image)
            
            # Convert the PyTorch tensor back to PIL Image
            transformed_image_pil = transforms.ToPILImage()(transformed_image)
            
            # Save the transformed image
            transformed_image_pil.save(os.path.join(augmented_dataset_path, f'{file.split(".")[0]}_transformed.jpg'))


                
# Create CSV file
# Path to your dataset (including augmented images)
dataset_path = '/home/samuele/Documenti/GitHub/Back_Squats_IPF/Men'
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

# Define the transformation to convert PIL images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

from torchvision.transforms.functional import resize
def resize_transform(img):
    return resize(img, size=(224,224))

dataset = datasets.ImageFolder(root=dataset_path, transform=resize_transform)

# Function to visualize images
def visualize_transformations(dataset, index):
    img, label = dataset[index]
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.title(f'Label: {dataset.classes[label]}')
    plt.show()

# # Visualize the transformations for the first ten images in the dataset
# for i in range(10):
#     visualize_transformations(dataset, i)  

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

from torchvision import transforms
from PIL import Image

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ConvNet architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Create an instance of the ConvNet
model = ConvNet()

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
n_epochs = 15

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
        loss.requires_grad = True

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

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms

current_dir = os.getcwd()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly change the brightness and contrast
    # transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
])

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

#visualize_transformations(dataset, 0)

# Now we set up the training and validation data loaders
from torch.utils.data import DataLoader, random_split

# Define the batch size
batch_size = 32

# Set the random seed
random_seed = 42

# Create the training and validation datasets
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

# Create the training and validation data loaders making sure that the data is tensors and not PIl images
# Create the training and validation data loaders
# Start the training loop
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    # Replace label strings with label numeric values
    target_map = {'valid': 0, 'above': 1, 'parallel': 2}
    target = [target_map[label] for label in target]

    return torch.stack(data), torch.tensor(target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze the parameters
for param in model.parameters():
    param.requires_grad = False


#Three outputs: above parallel, parallel, below parallel
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)

# Transfer the model to the GPU
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Define the number of epochs
n_epochs = 50

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize lists to store losses for plotting
train_losses = []
val_losses = []

# Initialize the best validation accuracy and early stopping parameters
best_val_accuracy = 0
patience = 10  # Number of epochs to wait for improvement
counter = 0  # Counter for early stopping

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

    # Check if validation accuracy has improved
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        counter = 0  # Reset the counter since there's an improvement
        # Save the model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1  # Increment the counter if there's no improvement

    # Append losses for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Step the learning rate scheduler
    scheduler.step()
    # Print metrics
    print(f'Epoch: {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.4f}')

    # Check for early stopping
    if counter >= patience:
        print("Early stopping. No improvement in validation accuracy.")
        break

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
torch.save(model.state_dict(), 'model.pt')

# Now we load the model
model.load_state_dict(torch.load('model.pt'))

# Plot training and validation losses
plt.figure(figsize=(10, 7))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

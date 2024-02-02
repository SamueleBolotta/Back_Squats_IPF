import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from utils import CustomImageDataset
from torch.utils.data import DataLoader, random_split
from utils import custom_collate, plot_losses, visualize_transformations
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from torchvision import datasets

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),  # Apply random perspective transformation
    transforms.RandomGrayscale(p=0.1),  # Convert images to grayscale with a probability of 0.1
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly change the brightness and contrast
    # transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
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

# get the current working directory
current_dir = os.getcwd()

# Path to your dataset
dataset_path = f'{current_dir}/dataset/'
# Path to csv dataframe file
df_file = f'{current_dir}/image_labels.csv'
dataset = CustomImageDataset(df_file, dataset_path, transform=transform)

# Define the batch size
batch_size = 32

# Set the random seed
random_seed = 42

# Create the training and validation datasets
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

# Load the training and validation data in batches 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


# Path to the Non-Athletes folder
non_athletes_path = f'{current_dir}/dataset2/Non-Athletes'

# Load Non-Athletes images as a separate dataset
non_athletes_dataset = datasets.ImageFolder(root=non_athletes_path, transform=transform)

# To append the Non-Athletes images to the test_dataset, we'll concatenate the datasets.

# Custom method to combine datasets
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        # Find which dataset the index falls in
        for d in self.datasets:
            if i < len(d):
                return d[i]
            i -= len(d)

# Combine the existing test_dataset with the non_athletes_dataset
combined_test_dataset = CombinedDataset(test_dataset, non_athletes_dataset)

# Now use the combined dataset for the test_loader
test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

#######################################################################################################################
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
# Modify the classifier head to match the number of classes in your dataset
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Add dropout for regularization
    nn.Linear(256, 3)  # Assuming you have 3 classes
)

# Transfer the model to the GPU
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)

# Define the number of epochs
n_epochs = 50

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


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
                                                                                                                        
test_loss = running_loss / len(test_dataset)
test_accuracy = correct_predictions / len(test_dataset)
                                                                                                                           
# Print the test metrics
print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}')

# # Now we save the model
torch.save(model.state_dict(), 'model.pt')

# Now we load the model
model.load_state_dict(torch.load('model.pt'))

# Call the function with the provided lists of train_losses and val_losses
plot_losses(train_losses, val_losses)

from sklearn.metrics import precision_score, recall_score, f1_score

# Inside your test loop, calculate precision, recall, and F1-score
predictions = torch.argmax(output, dim=1).cpu().numpy()
targets = target.cpu().numpy()

precision = precision_score(targets, predictions, average='weighted')
recall = recall_score(targets, predictions, average='weighted')
f1 = f1_score(targets, predictions, average='weighted')

# Print the additional metrics
print(f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}')

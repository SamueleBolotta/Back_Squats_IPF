import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from utils.utils import CustomImageDataset, build_rgb_dataset, save_images
from torch.utils.data import DataLoader, random_split
from utils.utils import custom_collate, plot_losses, visualize_transformations
import os
from PIL import Image
import pandas as pd
from torchvision import datasets
from tqdm import tqdm

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

image_labels = []
current_dir = os.path.abspath(os.getcwd()) # current working directory
original_dataset = f"{current_dir}/dataset"
train, test = build_rgb_dataset(original_dataset, train_ratio=0.8)
save_images(train, "train") # saved to path f"{current_dir}/dataset/{sex}/train"
save_images(test, "test") # saved to path f"{current_dir}/dataset/{sex}/test"

# create new augmented dataset saved to f"{current_dir}/dataset/{sex}/train_augmented"
for sex in ("Men", "Women"):
    # Path to your original dataset
    original_dataset_path = f'{current_dir}/dataset/{sex}/train'
    # Path to save augmented images
    augmented_dataset_path = f'{current_dir}/dataset/{sex}/train_augmented'

    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path)

    # Traverse the directory structure
    ## Pay attention to the fact that the images are in format RGBA, which is not supported by JPEG
    ## Convert it to RGB before saving
    for root, dirs, files in os.walk(original_dataset_path):
        # The subfolder name is the label
        label = os.path.basename(root)

        for file in tqdm(files):
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
                            
                        # Copy the original image to the augmented dataset
                        original_img_dest = os.path.join(path_with_label, file)
                        shutil.copy2(file_path, original_img_dest)

new_labels = {'Frontal_Above_parallel':'above', 'Lateral_Above_parallel':'above',
              'Frontal_Parallel':'parallel', 'Lateral_Parallel':'parallel',
              'Frontal_Valid':'valid', 'Lateral_Valid':'valid',
              'Frontal - Above parallel':'above', 'Lateral - Above parallel':'above',
              'Frontal - Parallel':'parallel', 'Lateral - Parallel':'parallel',
              'Frontal - Valid':'valid', 'Lateral - Valid':'valid'}

# Save the DataFrame to a CSV file
train_df = pd.DataFrame(image_labels, columns=["image", "label", "relative path"])
train_df['label'] = train_df['label'].replace(new_labels)
train_df.to_csv('image_labels.csv', index=False)

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
train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator1)

# Load the training and validation data in batches 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
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

# preprocess the test dataset for evaluation
print("PREPROCESSING TEST DATASET FOR EVALUATION...")
test_image_labels = []
for sex in ("Men", "Women"):
    # Path to your original dataset
    original_dataset_path = f'{current_dir}/dataset/{sex}/test'

    # Traverse the directory structure
    ## Pay attention to the fact that the images are in format RGBA, which is not supported by JPEG
    ## Convert it to RGB before saving
    for root, dirs, files in os.walk(original_dataset_path):
        # The subfolder name is the label
        label = os.path.basename(root)

        for file in tqdm(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                # avoid using images that are unlabelled
                if not "Screenshot" in file:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, f'{current_dir}/dataset/')
                    test_image_labels.append((file, label, relative_path))

# Save the DataFrame to a CSV file
df_test = pd.DataFrame(test_image_labels, columns=["image", "label", "relative path"])
df_test['label'] = df_test['label'].replace(new_labels)
df_test.to_csv('test_image_labels.csv', index=False)
df_test_file = f'{current_dir}/test_image_labels.csv'

# Define transformations for testing (no random transformations)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = CustomImageDataset(df_test_file, dataset_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

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

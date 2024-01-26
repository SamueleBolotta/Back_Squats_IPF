import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

def collect_images_labels(dataset_path, label_conversion):
    images = []
    labels = []

    # Map string labels to numeric indices
    label_to_index = {'above': 0, 'parallel': 1, 'valid': 2}

    for sex in ['Men', 'Women']:
        sex_path = os.path.join(dataset_path, sex, 'Original')
        if os.path.exists(sex_path):
            for weight_category_folder in os.listdir(sex_path):
                weight_category_path = os.path.join(sex_path, weight_category_folder)
                if os.path.isdir(weight_category_path):
                    for weight_class in os.listdir(weight_category_path):
                        weight_class_path = os.path.join(weight_category_path, weight_class)
                        if os.path.isdir(weight_class_path):
                            for label_folder in os.listdir(weight_class_path):
                                label_path = os.path.join(weight_class_path, label_folder)
                                if os.path.isdir(label_path):
                                    for image_name in os.listdir(label_path):
                                        if image_name.endswith(('.jpg', '.png')):
                                            image_path = os.path.join(label_path, image_name)
                                            images.append(image_path)
                                            # Convert string label to numeric index
                                            str_label = label_conversion.get(label_folder, label_folder)
                                            numeric_label = label_to_index[str_label]
                                            labels.append(numeric_label)
        else:
            print(f"Directory not found: {sex_path}")

    return images, labels

# New labels dictionary
new_labels = {
    'Frontal_Above_parallel': 'above', 'Lateral_Above_parallel': 'above',
    'Frontal_Parallel': 'parallel', 'Lateral_Parallel': 'parallel',
    'Frontal_Valid': 'valid', 'Lateral_Valid': 'valid',
    'Frontal - Above parallel': 'above', 'Lateral - Above parallel': 'above',
    'Frontal - Parallel': 'parallel', 'Lateral - Parallel': 'parallel',
    'Frontal - Valid': 'valid', 'Lateral - Valid': 'valid'
}

# Relative path to the dataset
dataset_path = './dataset'

# Use the function with new label conversion
images, labels = collect_images_labels(dataset_path, new_labels)
print(f"Total images loaded: {len(images)}")

# Train and Validation transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Test transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Calculate sizes for train, val, test
total_size = len(images)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Splitting the dataset
train_images, val_images, test_images = random_split(list(zip(images, labels)), [train_size, val_size, test_size])

print(f"Training set size: {len(train_images)}")
print(f"Validation set size: {len(val_images)}")
print(f"Test set size: {len(test_images)}")

# Create dataset instances with appropriate transforms
train_dataset = CustomImageDataset([i[0] for i in train_images], [i[1] for i in train_images], transform=train_transforms)
val_dataset = CustomImageDataset([i[0] for i in val_images], [i[1] for i in val_images], transform=train_transforms)
test_dataset = CustomImageDataset([i[0] for i in test_images], [i[1] for i in test_images], transform=test_transforms)

# Define the batch size and create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')
#Three outputs: above parallel, parallel, below parallel
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze the parameters
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)


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
# Initialize lists to store accuracies for plotting
train_accuracies = []
val_accuracies = []

# Initialize the best validation accuracy and early stopping parameters
best_val_accuracy = 0
patience = 10  # Number of epochs to wait for improvement
counter = 0  # Counter for early stopping

for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    total_train_loss = 0
    total_train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train_correct += (predicted == labels).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    train_accuracies.append(train_accuracy)

    print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val_correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

    scheduler.step()  # Adjust the learning rate

import matplotlib.pyplot as plt
# Plotting Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode
total_test_loss = 0
total_test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test_correct += (predicted == labels).sum().item()

avg_test_loss = total_test_loss / len(test_loader)
test_accuracy = total_test_correct / len(test_loader.dataset)

print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

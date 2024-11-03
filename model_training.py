import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Augmentation and Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define data directory
data_dir = r'data\images'  # Update this path as necessary

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

# Split dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load ResNet50 model and modify final layer
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Early Stopping Parameters
early_stop_count = 0
early_stop_limit = 3
best_val_loss = float('inf')

# Training function with validation and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    global early_stop_count, best_val_loss  # Access global variables

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Loop
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {time.time() - start_time:.2f}s")

        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # Save the best model
            torch.save(model.state_dict(), r'C:\Users\vijis\OneDrive\Desktop\project\BioBotanica\models\resnet50.pth')
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_limit:
                print("Early stopping triggered")
                break

# Train the model with early stopping
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Define load_model for loading the trained model in app.py
def load_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(r'C:\Users\vijis\OneDrive\Desktop\project\BioBotanica\models\resnet50.pth'))
    model.eval()  # Set to evaluation mode
    return model

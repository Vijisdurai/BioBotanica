import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define the path to your dataset
data_dir = r'data\images'

# Define transformations for data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize all images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),       # Random horizontal flip
    transforms.RandomRotation(20),                # Random rotation up to 20 degrees
    transforms.ColorJitter(brightness=0.2,        # Random brightness adjustment
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.ToTensor(),                        # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406],   # Normalization for ResNet models
                         [0.229, 0.224, 0.225])
])

# Load the dataset with transformations
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check the class names
class_names = dataset.classes
print("Class names:", class_names)

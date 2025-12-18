import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Increase PIL image size limit to avoid decompression bomb errors
Image.MAX_IMAGE_PIXELS = None

def get_data_loaders(train_dir, test_dir, batch_size=32):
    # Define transformations: resize, tensor conversion, normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_data.classes

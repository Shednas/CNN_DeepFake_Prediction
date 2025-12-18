"""CNN Model Architecture for Deepfake Detection.

A 3-layer convolutional neural network designed for binary classification
of images as real or AI-generated.

Input: 128x128 RGB images
Output: 2-class logits (AI-Generated, Real)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """Simple CNN for binary image classification.
    
    Architecture:
        - Conv Block 1: 3 → 16 channels (3×3 kernel)
        - Conv Block 2: 16 → 32 channels (3×3 kernel)
        - Conv Block 3: 32 → 64 channels (3×3 kernel)
        - FC Layer 1: 64×16×16 → 128 neurons (ReLU)
        - FC Layer 2: 128 → num_classes logits
    
    Each conv block includes max pooling (2×2) for downsampling.
    Input is downsampled from 128×128 → 64×64 → 32×32 → 16×16.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize CNN layers.
        
        Args:
            num_classes: Number of output classes (default: 2)
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional blocks with max pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv block 1: 128×128 → 64×64
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2: 64×64 → 32×32
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3: 32×32 → 16×16
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64×16×16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

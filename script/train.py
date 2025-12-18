import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from dataset_loader import get_data_loaders
from cnn_model import SimpleCNN

# Settings
train_dir = 'CNN_DeepFake_Prediction/dataset/train'
test_dir = 'CNN_DeepFake_Prediction/dataset/test'
batch_size = 32
epochs = 5
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
train_loader, test_loader, classes = get_data_loaders(
    train_dir, test_dir, batch_size
)

# Initialize model, loss, optimizer
model = SimpleCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# History storage
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': []
}

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = correct / total

    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

# Save model
torch.save(model.state_dict(), 'CNN_DeepFake_Prediction/model/cnn_model.pth')

# Save history
with open('CNN_DeepFake_Prediction/model/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print("Model and training history saved.")

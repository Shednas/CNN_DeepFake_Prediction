import torch
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset_loader import get_data_loaders
from cnn_model import SimpleCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data and model
train_loader, test_loader, classes = get_data_loaders(
    'CNN_DeepFake_Prediction/dataset/train',
    'CNN_DeepFake_Prediction/dataset/test'
)

model = SimpleCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load('CNN_DeepFake_Prediction/model/cnn_model.pth', map_location=device))
model.eval()

# Evaluation
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Metrics
acc = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true, y_pred, target_names=classes, output_dict=True
)
cm = confusion_matrix(y_true, y_pred)

# Print results
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
print("Confusion Matrix:")
print(cm)

# Save evaluation results
results = {
    'accuracy': acc,
    'classification_report': report,
    'confusion_matrix': cm,
    'classes': classes
}

with open('CNN_DeepFake_Prediction/model/test_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Test results saved to CNN_DeepFake_Prediction/model/test_results.pkl")

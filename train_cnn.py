import os
import argparse
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Define network architecture matching the notebook exactly
class FERNet(nn.Module):
    def __init__(self, num_classes=7):
        super(FERNet, self).__init__()
        # Input channel is 1 (grayscale 48x48 pixels)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 11 * 11, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def export_onnx(model_path, onnx_dir):
    print("Exporting model to ONNX format...")
    model = FERNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    dummy_input = torch.randn(1, 1, 48, 48)
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved successfully to {onnx_path}.")

    # Generate config files for Hugging Face Transformers.js
    config = {
        "architectures": ["FERNetForImageClassification"],
        "id2label": {
            "0": "angry",
            "1": "disgust",
            "2": "fear",
            "3": "happy",
            "4": "neutral",
            "5": "sad",
            "6": "surprise"
        },
        "label2id": {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6
        },
        "model_type": "cnn",
        "num_channels": 1
    }
    
    preprocessor_config = {
        "do_convert_rgb": False,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5],
        "image_std": [0.5],
        "rescale_factor": 0.00392156862745098,
        "size": {
            "height": 48,
            "width": 48
        }
    }

    with open(os.path.join(onnx_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(onnx_dir, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f, indent=2)
    print("Metadata config files created in ONNX directory.")

def main():
    parser = argparse.ArgumentParser(description="Train FERNet PyTorch CNN model on local dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("-l", "--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--onnx-only", action="store_true", help="Only export existing pth to ONNX.")
    args = parser.parse_args()

    model_path = "cnn_model.pth"
    onnx_dir = "public/cnn_onnx"

    if args.onnx_only:
        if os.path.exists(model_path):
            export_onnx(model_path, onnx_dir)
        else:
            print(f"Error: checkpoint {model_path} not found.")
        return

    # Select acceleration backend
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU execution.")

    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found. Please verify the symlink.")
        return

    # Transformations matching the notebook
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Total training images: {len(train_dataset)} | Classes: {train_dataset.class_to_idx}")
    print(f"Total testing images: {len(test_dataset)}")

    # Initialize model
    model = FERNet(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Resume training if checkpoint exists
    start_epoch = 0
    if os.path.exists(model_path):
        print(f"Resuming training from checkpoint: {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Checkpoint weights loaded successfully.")

    print("\nTraining CNN Model started...")
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct / total

        # Validation Phase
        model.eval()
        running_test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_acc = 100. * test_correct / test_total
        
        print(f"Epoch [{epoch}/{start_epoch + args.epochs}] -> "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

        # Save checkpoint and export to ONNX at the end of every epoch for live updates in the browser
        torch.save(model.state_dict(), model_path)
        export_onnx(model_path, onnx_dir)

if __name__ == "__main__":
    main()

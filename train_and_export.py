import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Define the exact CNN architecture from MNIST_ (1).ipynb
# Keras equivalent:
# Conv2D(32, (3,3), relu) -> MaxPool(2,2) -> Conv2D(64, (3,3), relu) -> MaxPool(2,2) -> Flatten(1600) -> Dense(128, relu) -> Dense(10, softmax)

class MNISTNotebookCNN(nn.Module):
    def __init__(self):
        super(MNISTNotebookCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def train_model():
    print("[*] Setting up MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)) # Scale to [0, 1] matching notebook x_train / 255.0
    ])

    data_dir = "./data"
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device("cpu")
    model = MNISTNotebookCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("[*] Training CNN Model matching MNIST_ (1).ipynb for 3 epochs...")
    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = 100.0 * correct / total
        print(f"Epoch {epoch}/3 - Loss: {running_loss/len(train_loader):.4f} - Test Accuracy: {acc:.2f}%")

    # Save torch model
    torch.save(model.state_dict(), "mnist_notebook_cnn.pth")
    print("[+] Model saved to mnist_notebook_cnn.pth")

    # Export weights to JSON for JavaScript frontend inference
    weights = {
        "conv1_w": model.conv1.weight.detach().cpu().numpy().tolist(), # [32, 1, 3, 3]
        "conv1_b": model.conv1.bias.detach().cpu().numpy().tolist(),   # [32]
        "conv2_w": model.conv2.weight.detach().cpu().numpy().tolist(), # [64, 32, 3, 3]
        "conv2_b": model.conv2.bias.detach().cpu().numpy().tolist(),   # [64]
        "fc1_w": model.fc1.weight.detach().cpu().numpy().tolist(),     # [128, 1600]
        "fc1_b": model.fc1.bias.detach().cpu().numpy().tolist(),       # [128]
        "fc2_w": model.fc2.weight.detach().cpu().numpy().tolist(),     # [10, 128]
        "fc2_b": model.fc2.bias.detach().cpu().numpy().tolist()        # [10]
    }

    os.makedirs("canvas", exist_ok=True)
    os.makedirs("mnist_terminal_classifier/canvas", exist_ok=True)
    
    with open("canvas/model_weights.json", "w") as f:
        json.dump(weights, f)
    with open("mnist_terminal_classifier/canvas/model_weights.json", "w") as f:
        json.dump(weights, f)

    print("[+] Exported model weights to canvas/model_weights.json successfully!")

if __name__ == "__main__":
    train_model()

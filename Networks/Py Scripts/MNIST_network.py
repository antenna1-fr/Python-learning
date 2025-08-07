import torch
import torchvision
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Networks import config
import sklearn as sk
import pandas as pd
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load MNIST as tensors
def preprocess_mnist(dataset):
    data = dataset.data.float().div(255.0).unsqueeze(1)  # scale to [0, 1]
    targets = dataset.targets
    data = data.to(device)
    targets = targets.to(device)
    dataset = TensorDataset(data, targets)
    return torch.utils.data.TensorDataset(data, targets)

# Download + preprocess once
raw_train = datasets.MNIST(root=config.DATA_DIR, train=True, download=True)
raw_test = datasets.MNIST(root=config.DATA_DIR, train=False, download=True)

train_dataset = preprocess_mnist(raw_train)
test_dataset = preprocess_mnist(raw_test)
# Set hyperparameters

learning_rate = .02
total_epochs = 50
batch_size = 5000
dropout_probability = 0.2

print(f'training on device: {device}')
# Create Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
# Create Model
class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, 64),
            nn.Hardswish(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
model = MNISTnet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_array = []
# Training loop
def training_loop():
    last_avg = 0
    for epoch in range(1, total_epochs+1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_array.append(loss.item())
        if epoch % 10 == 0:
            avg = epoch_loss / len(train_loader)
            if abs(last_avg-avg) <= .04 and avg <= .05:    # Early stopping if loss is no longer improving
                print(f'difference is {abs(last_avg-avg)}')
                print(f"Early stopping at epoch {epoch:5d}")
                break
            print(f"Epoch {epoch:5d} — avg batch loss: {avg:.4f}")
            last_avg = avg

import cProfile

cProfile.run('training_loop()', sort= 'tottime')
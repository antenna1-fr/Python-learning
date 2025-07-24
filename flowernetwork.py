import torch
import sklearn as sk
import pandas as pd
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader

df = load_iris(as_frame=True)
raw_data = {}
for key in df.keys():
    if key in ['data', 'target']:
        raw_data[key] = df[key]
print(raw_data)

df.frame['target'] = df.target_names[df.target]

print(df['target'].describe())
print(df['data'].describe())
_ = sns.pairplot(df.frame, hue='target')
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(raw_data['data']))

from sklearn.model_selection import train_test_split
X_raw = raw_data['data']
y_raw = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(X_raw,y_raw, test_size=0.2, random_state=42)
learning_rate = .02
total_epochs = 30000
batch_size = 60
dropout_probability = 0.3


X_train = torch.tensor(X_train.to_numpy(), dtype = torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype = torch.long)
X_test  = torch.tensor(X_test.to_numpy(), dtype = torch.float32)
y_test  = torch.tensor(y_test.to_numpy(), dtype = torch.long)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Define the neural network model
class Flowernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(8,3),
        )
    def forward(self, x):
        return self.net(x)
    
model = Flowernet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

last_avg = 0.0
loss_array = []
# Training loop
for epoch in range(1, total_epochs + 1):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10  == 0:
        loss_array.append(loss.item())
    if epoch % 1000 == 0:
        avg = epoch_loss / len(train_loader)
        if abs(last_avg-avg) <= 5e-4 or avg <= .06:    # Early stopping if loss is no longer improving
            print(f'difference is {abs(last_avg-avg)}')
            print(f"Early stopping at epoch {epoch:5d}")
            break
        print(f"Epoch {epoch:5d} — avg batch loss: {avg:.4f}")
        last_avg = avg
    


from sklearn.metrics import confusion_matrix, classification_report
model.eval()
with torch.no_grad():
    out = model(X_test)
    predicted_labels = torch.argmax(out, dim=1).flatten()
    true_labels = y_test

    # Calculate accuracy
    accuracy = (predicted_labels == true_labels).sum().item() / len(y_test) * 100
    print(f"Accuracy: {accuracy:.4f}%")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=df.target_names)
    print("\nClassification Report:")
    print(class_report)

loss_array_epochs = [i * 10 for i in range(len(loss_array))]
for i in range (len(loss_array)):
    loss_array[i] = np.mean(loss_array[max(0,i-9):i+1])

plt.figure(figsize = (10,5))
plt.plot(loss_array_epochs, loss_array, marker = 'o', linestyle = '')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(False)
plt.show()

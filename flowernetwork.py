
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


# %%
df = load_iris(as_frame=True)
raw_data = {}
for key in df.keys():
    if key in ['data', 'target']:
        raw_data[key] = df[key]
print(raw_data)

df.frame['target'] = df.target_names[df.target]

print(df['target'].describe())
print(df['data'].describe())

# %%
_ = sns.pairplot(df.frame, hue='target')
plt.show()

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(raw_data['data']))


# %%
from sklearn.model_selection import train_test_split
X_raw = raw_data['data']
y_raw = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(X_raw,y_raw, test_size=0.2, random_state=42)

# %%
learning_rate = .001
total_epochs = 10000
batch_size = 60


# %%
X_train = torch.tensor(X_train.to_numpy(), dtype = torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype = torch.long)
X_test  = torch.tensor(X_test.to_numpy(), dtype = torch.float32)
y_test  = torch.tensor(y_test.to_numpy(), dtype = torch.long)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# %%
class Flowernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,3),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
    
model = Flowernet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

last_avg = 0.0
    



# %%
for epoch in range(1, total_epochs + 1):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 1000 == 0:
        avg = epoch_loss / len(train_loader)
        if abs(last_avg-avg) <= 1e-3:    # Early stopping if loss is no longer improving
            print(f'difference is {last_avg-avg}')
            print(f"Early stopping at epoch{epoch:5d}")
            break
        print(f"Epoch {epoch:5d} — avg batch loss: {avg:.4f}")
        last_avg = avg

# %%
with torch.no_grad():
    out = model(X_test)
    preds = (out > 0.5).int().squeeze()
    print(preds.flatten())
    print(y_test)


# %%


# %%




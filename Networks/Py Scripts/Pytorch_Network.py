import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([[0,1],[1,0],[0,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[1],[1],[0],[0]], dtype=torch.float32)

batch_size    = 2
learning_rate = 0.1
total_epochs  = 10000

dataset = TensorDataset(X,y)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class XORnet(nn.Module): # Creates a subclass of "nn.Module" named XORnet
    def __init__(self):
        super().__init__()  # Initializes the __init__ function of the parent class
        self.net = nn.Sequential( # Defines the neuron structure
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid()
        )
    def forward(self,x): # forward pass definition
        return self.net(x)
    
model = XORnet()

criterion = nn.BCELoss() # specifies loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # the descent function to use for weight adjustments
last_avg = 1

for epoch in range(1, total_epochs+1):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)           # forward pass
        loss = criterion(preds, yb) # compute deviation from true labels
        loss.backward()             # backpropagation
        optimizer.step()            # update weights
        epoch_loss += loss.item()   # accumulates loss

    if epoch % 1000 == 0:
        avg = epoch_loss / len(loader)
        if last_avg-avg <= 1e-4:    # Early stopping if loss is no longer improving
            print(f"Early stopping at epoch{epoch:5d}")
            break
        print(f"Epoch {epoch:5d} — avg batch loss: {avg:.4f}")
        last_avg = avg

with torch.no_grad():
    out = model(X)
    preds = (out > 0.5).int().squeeze()
    print("Final raw outputs:", out.squeeze().tolist())
    print("Final predictions:", preds.tolist())
    print("True predictions: ", y.squeeze().tolist())





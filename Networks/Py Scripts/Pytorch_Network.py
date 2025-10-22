import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
torch.set_num_threads(1)

# Data (vectorized full batch)
X = torch.tensor([[0.,1.],
                  [1.,0.],
                  [0.,0.],
                  [1.,1.]], dtype=torch.float32)
y = torch.tensor([[1.],[1.],[0.],[0.]], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,1)          # no Sigmoid here
        )
    def forward(self,x):
        return self.net(x)

model = XORNet()
criterion = nn.BCEWithLogitsLoss()   # combines Sigmoid + BCE
optimizer = optim.SGD(model.parameters(), lr=0.1)

total_epochs = 10000
last_loss = float('inf')

for epoch in range(1, total_epochs+1):
    # (Optional) shuffle with a single index permutation, no DataLoader
    idx = torch.randperm(X.size(0))
    xb, yb = X[idx], y[idx]

    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)            # single forward on full batch
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        curr = loss.detach().item()
        print(f"Epoch {epoch:5d} — loss: {curr:.6f}")
        if (last_loss - curr) <= 1e-4:
            print(f"Early stopping at epoch {epoch}")
            break
        last_loss = curr

with torch.no_grad():
    logits = model(X)
    probs  = torch.sigmoid(logits)
    preds  = (probs > 0.5).int().squeeze()
    print("Final raw outputs:", probs.squeeze().tolist())
    print("Final predictions:", preds.tolist())
    print("True predictions: ", y.squeeze().tolist())

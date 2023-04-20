import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

data = pd.read_csv('creditos.csv')
edades = torch.tensor(data['edad'].values, dtype=torch.float32)
salarios = torch.tensor(data['salario'].values, dtype=torch.float32)
labels = torch.tensor(data['apta_para_credito'].values, dtype=torch.float32)

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    
model = Modelo()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    inputs = torch.stack([edades, salarios], dim=1)
    outputs = model(inputs)
    # reshape the target tensor to have the same size as the output tensor
    reshaped_labels = labels.view(-1, 1)
    loss = criterion(outputs, reshaped_labels)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    inputs = torch.tensor([30.0, 50000.0])
    output = model(inputs)
    print(output.item())

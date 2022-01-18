import torch
from torch import nn
import numpy as np 
from typing import Dict

input = np.array([[1,1],
                  [1,0], 
                  [0,1], 
                  [0,0]])


target = np.array([[0],[1],[1],[0]])


def convert_to_tensor(input: np.ndarray, target: np.ndarray)->Dict[np.ndarray, np.ndarray]:
    """ Convert from np array to tensor """
    X = torch.from_numpy(input.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32))

    x_train = X[:80, :]
    x_test = X[80:, :]
    y_train = y[:80, :]
    y_test = y[80:, :]


    dataset = {
        'X' : X,
        'y' : y,
    }

    return dataset

dataset = convert_to_tensor(input, target)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
        )

    def forward(self, X):
        y_pred = self.fc(X)
        y_pred = torch.sigmoid(y_pred)
        return y_pred


model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 200   

for epoch in range(epochs):
    y_pred = model(dataset['X'])
    loss = criterion(y_pred, dataset['y'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(dataset['X'])
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(dataset['y']).sum() / float(dataset['y'].shape[0])
    print(f'accuracy: {acc.item():.4f}')


        

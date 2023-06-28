import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load data
data = {}

with open('data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row

    for row in reader:
        c_rate = float(row[0])
        index = int(row[1])
        value = float(row[2])

        if c_rate not in data:
            data[c_rate] = []

        data[c_rate].append((index, value))

c_rates = list(data.keys())
results = []

for c_rate in c_rates:
    c_rate_data = data[c_rate]
    x_data = [item[0] for item in c_rate_data]
    y_data = [item[1] for item in c_rate_data]

    x_data = Variable(torch.tensor(x_data).float())
    y_data = Variable(torch.tensor(y_data).float())

    print(x_data, y_data)

    # Define model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.A = nn.Parameter(torch.tensor(0.0001))
            self.B = nn.Parameter(torch.tensor(0.1))
            self.C = nn.Parameter(torch.tensor(0.001))

        def forward(self, x):
            return 1 - self.A * torch.sqrt(x) - self.B * torch.exp(self.C * x) + self.B

    net = Net()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    # Train model
    for t in range(10000):
        prediction = net(x_data)
        loss = loss_func(prediction, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    A, B, C = net.A.item(), net.B.item(), net.C.item()
    results.append([c_rate, A, B, C])

# Print results
for result in results:
    print(f'c-rate: {result[0]}, A: {result[1]}, B: {result[2]}, C: {result[3]}')

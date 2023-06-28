import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import csv
import matplotlib.pyplot as plt

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


# Define new model
class ABCModel(nn.Module):
    def __init__(self):
        super(ABCModel, self).__init__()
        self.fc1 = nn.Linear(1, 3)

    def forward(self, x):
        x = self.fc1(x)
        return x


abc_model = ABCModel()

# Prepare data for new model
x_data = []
y_data = []

for result in results:
    c_rate, A, B, C = result
    x_data.append([c_rate])
    y_data.append([A, B, C])

x_data = Variable(torch.tensor(x_data).float())
y_data = Variable(torch.tensor(y_data).float())

# Define optimizer and loss function for new model
optimizer = torch.optim.Adam(abc_model.parameters(), lr=0.001)
loss_func = nn.MSELoss()

# Train new model
for t in range(10000):
    prediction = abc_model(x_data)
    loss = loss_func(prediction, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test new model
test_c_rate = Variable(torch.tensor([[0.7]]).float())
test_prediction = abc_model(test_c_rate)
print(
    f'c-rate: 0.7, A: {test_prediction[0][0].item()}, B: {test_prediction[0][1].item()}, C: {test_prediction[0][2].item()}')

# # Plot data
# for c_rate in c_rates:
#     c_rate_data = data[c_rate]
#     x_data = [item[0] for item in c_rate_data]
#     y_data = [item[1] for item in c_rate_data]
#
#     plt.scatter(x_data, y_data, label=f'c-rate: {c_rate}')
#
# # Plot predictions
# x_range = range(0, 1001)
# for result in results:
#     c_rate, A, B, C = result
#     y_pred = [1 - A * np.sqrt(x) - B * np.exp(C * x) + B for x in x_range]
#     plt.plot(x_range, y_pred, label=f'c-rate: {c_rate}')
#
# plt.legend()
# plt.show()


fig, axs = plt.subplots(len(c_rates), figsize=(10, 20))

for i, c_rate in enumerate(c_rates):
    ax = axs[i]
    ax.set_title(f'c-rate: {c_rate}')

    # Plot data
    c_rate_data = data[c_rate]
    x_data = [item[0] for item in c_rate_data]
    y_data = [item[1] for item in c_rate_data]

    ax.scatter(x_data, y_data)

    # Plot prediction
    x_range = range(0, 1001)
    result = [result for result in results if result[0] == c_rate][0]
    c_rate, A, B, C = result
    y_pred = [1 - A * np.sqrt(x) - B * np.exp(C * x) + B for x in x_range]
    ax.plot(x_range, y_pred)

plt.show()
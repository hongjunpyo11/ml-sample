# Define new model
class ABCModel(nn.Module):
    def __init__(self):
        super(ABCModel, self).__init__()
        self.fc1 = nn.Linear(6, 3)

    def forward(self, x):
        x = self.fc1(x)
        return x

abc_model = ABCModel()

# Prepare data for new model
x_data = []
y_data = []

for result in results:
    c_rate, A, B, C = result
    x_data.append([c_rate, 25, 1.00, 98.0, 92.0, 0.15])  # Add all conditions here
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
test_c_rate = Variable(torch.tensor([[0.7, 25, 1.00, 98.0, 92.0, 0.15]]).float())  # Add all conditions here
test_prediction = abc_model(test_c_rate)
print(f'c-rate: 0.7, A: {test_prediction[0][0].item()}, B: {test_prediction[0][1].item()}, C: {test_prediction[0][2].item()}')

많은 조건이 생긴다면
data = {
    (25, 1.00, 98.0, 92.0, 0.15): [(0, 1.0), (29, 0.975), (58, 0.965), (86, 0.96), ...]
}
이렇게 정해주고
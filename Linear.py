import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 준비 및 전처리
x_data_0_15 = torch.tensor([[0.], [29.], [58.], [86.]])
y_data_0_15 = torch.tensor([[1.0], [0.975], [0.965], [0.960]])

x_data_0_33 = torch.tensor([[0.], [17.], [35.], [52.]])
y_data_0_33 = torch.tensor([[1.0], [97.2], [96.3], [95.8]])

x_data_0_50 = torch.tensor([[0.], [14.], [29.], [43.]])
y_data_0_50 = torch.tensor([[1.0], [0.974], [0.968], [0.956]])


# 모델 정의
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.A = nn.Parameter(torch.tensor(0.0001))  # A를 모델의 파라미터로 정의
        self.B = nn.Parameter(torch.tensor(0.1))  # B를 모델의 파라미터로 정의
        self.C = nn.Parameter(torch.tensor(0.0001))  # C를 모델의 파라미터로 정의

    def forward(self, x):
        return 1 - self.A * torch.sqrt(x) - self.B * torch.exp(self.C * x) + self.B


# 손실 함수 정의
criterion = nn.MSELoss()

# 모델 및 옵티마이저 초기화
model_0_15 = PolynomialRegressionModel()
optimizer_0_15 = optim.SGD(model_0_15.parameters(), lr=0.001)

model_0_33 = PolynomialRegressionModel()
optimizer_0_33 = optim.SGD(model_0_33.parameters(), lr=0.001)

model_0_50 = PolynomialRegressionModel()
optimizer_0_50 = optim.SGD(model_0_50.parameters(), lr=0.001)

# 학습 과정
num_epochs = 1000

# c-rate 0.15 학습
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_0_15(x_data_0_15)
    loss = criterion(outputs, y_data_0_15)

    # Backward pass 및 가중치 업데이트
    optimizer_0_15.zero_grad()
    loss.backward()
    optimizer_0_15.step()

# c-rate 0.33 학습
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_0_33(x_data_0_33)
    loss = criterion(outputs, y_data_0_33)

    # Backward pass 및 가중치 업데이트
    optimizer_0_33.zero_grad()
    loss.backward()
    optimizer_0_33.step()

# c-rate 0.50 학습
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_0_50(x_data_0_50)
    loss = criterion(outputs, y_data_0_50)

    # Backward pass 및 가중치 업데이트
    optimizer_0_50.zero_grad()
    loss.backward()
    optimizer_0_50.step()

# A, B, C 값을 확인
print('Estimated A (c-rate 0.15):', model_0_15.A.item())
print('Estimated B (c-rate 0.15):', model_0_15.B.item())
print('Estimated C (c-rate 0.15):', model_0_15.C.item())

print('Estimated A (c-rate 0.33):', model_0_33.A.item())
print('Estimated B (c-rate 0.33):', model_0_33.B.item())
print('Estimated C (c-rate 0.33):', model_0_33.C.item())

print('Estimated A (c-rate 0.50):', model_0_50.A.item())
print('Estimated B (c-rate 0.50):', model_0_50.B.item())
print('Estimated C (c-rate 0.50):', model_0_50.C.item())

c_rates = [0.15, 0.33, 0.50]
a_values = [model_0_15.A.item(), model_0_33.A.item(), model_0_50.A.item()]
b_values = [model_0_15.B.item(), model_0_33.B.item(), model_0_50.B.item()]
c_values = [model_0_15.C.item(), model_0_33.C.item(), model_0_50.C.item()]

# A 값의 트렌드 그래프
plt.plot(c_rates, a_values, marker='o')
plt.xlabel('c-rate')
plt.ylabel('A')
plt.title('Trend of A')
plt.show()

# B 값의 트렌드 그래프
plt.plot(c_rates, b_values, marker='o')
plt.xlabel('c-rate')
plt.ylabel('B')
plt.title('Trend of B')
plt.show()

# C 값의 트렌드 그래프
plt.plot(c_rates, c_values, marker='o')
plt.xlabel('c-rate')
plt.ylabel('C')
plt.title('Trend of C')
plt.show()

# c-rate 0.2에 대한 ABC 값 예측
a_0_2 = np.interp(0.2, c_rates, a_values)
b_0_2 = np.interp(0.2, c_rates, b_values)
c_0_2 = np.interp(0.2, c_rates, c_values)

# 결과 출력
print('Predicted A (c-rate 0.2):', a_0_2)
print('Predicted B (c-rate 0.2):', b_0_2)
print('Predicted C (c-rate 0.2):', c_0_2)

plt.plot(['A', 'B', 'C'], [a_0_2, b_0_2, c_0_2], marker='o')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.title('Predicted Values for c-rate 0.2')
plt.show()


# 모든 c-rate
plt.plot(c_rates, a_values, marker='o', label='A')
plt.plot(c_rates, b_values, marker='o', label='B')
plt.plot(c_rates, c_values, marker='o', label='C')
plt.xlabel('c-rate')
plt.ylabel('Value')
plt.title('Trends of A, B, C')
plt.legend()
plt.show()

# x 범위 설정
x_range = torch.arange(0, 50, 0.1)

# 예측값 계산
predictions_0_15 = model_0_15(x_range)
predictions_0_33 = model_0_33(x_range)
predictions_0_50 = model_0_50(x_range)

# 데이터셋 그래프
plt.scatter(x_data_0_15, y_data_0_15, label='Data')
# 예측 그래프
plt.plot(x_range, predictions_0_15.detach().numpy(), color='red', label='Predictions (c-rate 0.15)')
plt.plot(x_range, predictions_0_33.detach().numpy(), color='blue', label='Predictions (c-rate 0.33)')
plt.plot(x_range, predictions_0_50.detach().numpy(), color='green', label='Predictions (c-rate 0.50)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터셋 준비 및 전처리
x_data = torch.tensor([[0.], [13.], [25.]])
y_data = torch.tensor([100.0000, 97.0000, 96.0000])


# 모델 정의
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.A = nn.Parameter(torch.tensor(0.0))  # A를 모델의 파라미터로 정의
        self.B = nn.Parameter(torch.tensor(0.0))  # B를 모델의 파라미터로 정의
        self.C = nn.Parameter(torch.tensor(0.0))  # C를 모델의 파라미터로 정의

    def forward(self, x):
        return 1 - self.A * torch.sqrt(x) - self.B * torch.exp(self.C * x) + self.B


# 손실 함수 정의
criterion = nn.MSELoss()

# 모델 및 옵티마이저 초기화
model = PolynomialRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습 과정
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_data)
    loss = criterion(outputs, y_data)

    # Backward pass 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# A, B, C 값을 확인
print('Estimated A:', model.A.item())
print('Estimated B:', model.B.item())
print('Estimated C:', model.C.item())

# 그래프 생성
x_range = torch.arange(26, 50, 0.1)  # 25보다 큰 값으로 x 범위 설정
predictions = model(x_range)  # 모델의 예측값 계산

# 데이터셋 그래프
plt.scatter(x_data, y_data, label='Data')
# 예측 그래프
plt.plot(x_range, predictions.detach().numpy(), color='red', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

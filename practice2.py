import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import Adam
import pandas as pd
import csv
import numpy as np

result = []

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 첫 번째 줄은 헤더이므로 건너뜁니다.

    for row in reader:
        row_data = [float(item) for item in row]
        result.append(row_data)

c_rates = [0.15, 0.33, 0.50]
data = {}
for item in result:
    c_rate, index, value = item
    if c_rate not in data:
        data[c_rate] = []
    data[c_rate].append((index, value))
print(data)


# 비선형 회귀 모델 정의
class NonlinearRegressionModel(nn.Module):
    def __init__(self):
        super(NonlinearRegressionModel, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.001))
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.001))

    def forward(self, x):
        return 1 - self.a * torch.sqrt(x) - self.b * torch.exp(self.c * x) + self.b

# 학습 및 추론 함수
def train_model(c_rate):
    # 데이터 로드
    dataset = data[c_rate]
    x_data = torch.tensor([item[0] for item in dataset], dtype=torch.float32)
    y_data = torch.tensor([item[1] for item in dataset], dtype=torch.float32)

    # 모델 초기화
    model = NonlinearRegressionModel()

    # 손실 함수 정의
    criterion = nn.MSELoss()

    # 옵티마이저 정의
    optimizer = Adam(model.parameters(), lr=0.001)

    # 학습 과정
    num_epochs = 5000

    for epoch in range(num_epochs):
        # Forward 계산
        outputs = model(x_data)

        # 손실 계산
        loss = criterion(outputs, y_data)

        # Backward 및 옵티마이저 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 학습된 A, B, C 값 반환
    return model.a.item(), model.b.item(), model.c.item()

# 각 c-rate에 대해 모델 학습 및 A, B, C 값 추정
results = {}
for c_rate in c_rates:
    a, b, c = train_model(c_rate)
    results[c_rate] = {'A': a, 'B': b, 'C': c}

# 원하는 c-rate에 해당하는 A, B, C 값을 추출
target_c_rate = 0.33
target_a = results[target_c_rate]['A']
target_b = results[target_c_rate]['B']
target_c = results[target_c_rate]['C']
print(f"A: {target_a}, B: {target_b}, C: {target_c}")

# c-rate 0.33에 대한 데이터 가져오기
x_data = np.array([item[0] for item in data[target_c_rate]])
y_data = np.array([item[1] for item in data[target_c_rate]])

# 그래프 생성
x_range = np.linspace(min(x_data), max(x_data), 100)
y_predicted = 1 - target_a * np.sqrt(x_range) - target_b * np.exp(target_c * x_range) + target_b

# 그래프 그리기
plt.plot(x_data, y_data, 'ro', label='Actual')  # 실제 데이터
plt.plot(x_range, y_predicted, 'b-', label='Predicted')  # 예측값
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'c-rate {target_c_rate} - Actual vs Predicted')
plt.show()
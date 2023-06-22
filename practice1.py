import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import Adam
import pandas as pd
import csv

# data = []

# with open('data.csv', 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # 첫 번째 줄은 헤더이므로 건너뜁니다.

#     for row in reader:
#         row_data = [float(item) for item in row]
#         data.append(row_data)

# 데이터셋 정의
x_data = torch.tensor([0, 29, 58, 86, 114, 142, 170, 197, 224, 251, 279, 305, 332, 358, 384, 410,
                       0, 17, 35, 52, 69, 86, 103, 120, 136, 151, 167, 184, 200, 216, 232, 249,
                       265, 281, 296, 312, 328, 343, 359, 374, 389, 405, 420, 0, 14, 29, 43, 57,
                       71, 85, 99, 113, 127, 141, 154, 170, 184, 197, 211, 224, 237, 251, 264,
                       277, 290, 303, 316, 329, 342, 355, 367, 380, 393, 405, 418, 430, 443], dtype=torch.float32)
y_data = torch.tensor([1.00, 0.975, 0.965, 0.960, 0.952, 0.948, 0.943, 0.936, 0.931, 0.927, 0.920, 0.913,
                       0.908, 0.906, 0.900, 0.893, 1.00, 97.2, 96.3, 95.8, 95.5, 94.5, 0.939, 0.934,
                       0.932, 0.922, 0.909, 0.900, 0.898, 0.889, 0.886, 0.879, 0.875, 0.869, 0.862,
                       0.859, 0.858, 0.850, 0.848, 0.842, 0.842, 0.833, 0.826, 1.00, 0.974, 0.968,
                       0.956, 0.950, 0.946, 0.938, 0.933, 0.929, 0.921, 0.916, 0.910, 0.897, 0.883,
                       0.878, 0.875, 0.871, 0.864, 0.863, 0.858, 0.853, 0.849, 0.846, 0.842, 0.837,
                       0.836, 0.830, 0.828, 0.822, 0.819, 0.814, 0.812, 0.813, 0.803], dtype=torch.float32)

# 비선형 회귀 모델 정의
class NonlinearRegressionModel(nn.Module):
    def __init__(self):
        super(NonlinearRegressionModel, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.001))
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.001))

    def forward(self, x):
        return 1 - self.a * torch.sqrt(x) - self.b * torch.exp(self.c * x) + self.b

# 모델 초기화
model = NonlinearRegressionModel()

# 손실 함수 정의
criterion = nn.MSELoss()

# 옵티마이저 정의
optimizer = Adam(model.parameters(), lr=0.00001)

# 학습 과정
num_epochs = 100000

for epoch in range(num_epochs):
    # Forward 계산
    outputs = model(x_data)

    # 손실 계산
    loss = criterion(outputs, y_data)

    # Backward 및 옵티마이저 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습된 A, B, C 값 출력
print(f"A: {model.a.item()}, B: {model.b.item()}, C: {model.c.item()}")
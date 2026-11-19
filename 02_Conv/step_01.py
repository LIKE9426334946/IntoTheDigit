"""
没有激活函数和归一化函数，使用两层卷积函数，使用MSE得到损失
"""

import torch
import torch.nn as nn

torch.manual_seed(23)

# 1. 构造输入和真实标签
x = torch.arange(1, 49, dtype=torch.float32).reshape(1, 3, 4, 4)

target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])


# 2. 定义两层卷积网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, bias=False)

        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x.retain_grad()
        self.y = x

        x = self.conv2(x)
        x.retain_grad()
        self.z = x

        return x


model = CNN()

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 打印初始参数
print("========== 初始参数 ==========")

for name, param in model.named_parameters():
    print(f"\n{name}")
    print("形状:", param.shape)
    print("参数:", param.data)

# 5. 前向传播
prediction = model(x)
loss = criterion(prediction, target)

print("\n========== 前向传播 ==========")
print("预测结果:\n", prediction.detach())
print("真实结果:\n", target)
print("Loss:", loss.item())

# 6. 反向传播
optimizer.zero_grad()
loss.backward()

print("\n========== 参数梯度 ==========")

for name, param in model.named_parameters():
    print(f"\n{name}")
    print("梯度:", param.grad)

# 7. 保存更新前的参数
old_params = {name: param.detach().clone() for name, param in model.named_parameters()}

# 8. 更新参数
optimizer.step()

# 9. 对比更新前后
print("\n========== 参数更新 ==========")

for name, param in model.named_parameters():
    old = old_params[name]
    new = param.detach()

    print(f"\n{name}")
    print("更新前:\n", old)
    print("梯度:\n", param.grad)
    print("更新量:\n", new - old)
    print("更新后:\n", new)

    # 验证 SGD 更新公式
    expected = old - 0.1 * param.grad
    print("公式验证:", torch.allclose(new, expected))

# 10. 再次前向传播
with torch.no_grad():
    new_prediction = model(x)
    new_loss = criterion(new_prediction, target)

print("\n========== 更新后 ==========")
print("更新前 Loss:", loss.item())
print("更新后 Loss:", new_loss.item())

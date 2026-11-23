"""
卷积层雅可比矩阵测试
"""

import torch
import torch.nn as nn

torch.manual_seed(23)

x_1 = torch.randint(0, 10, (1, 2, 3, 3), dtype=torch.float32, requires_grad=True)
target = torch.rand((1, 2, 3, 3))
conv = nn.Conv2d(2, 2, 3, padding=1)
x_2 = conv(x_1)
x_2.retain_grad()
print("x_2的形状为", x_2.shape)

criterion = nn.MSELoss()
loss = criterion(x_2, target)
loss.backward()
J = torch.autograd.functional.jacobian(lambda z: conv(z), x_1)

print("雅可比矩阵形状为：", J.shape)


print("结束")

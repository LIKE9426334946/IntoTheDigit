"""
看看残差连接梯度是如何传播的，没有relu函数
"""

import torch
import torch.nn as nn

torch.manual_seed(23)

x_1 = torch.randint(0, 10, (2, 3), dtype=torch.float32, requires_grad=True)
linear = nn.Linear(3, 4, bias=False)
x_2 = linear(x_1)
x_2.retain_grad()
print("x_2的形状为", x_2.shape)

target = torch.randint(0, 10, (2, 4)).float()
criterion = nn.MSELoss()
loss = criterion(x_2, target)

grad = torch.autograd.grad(outputs=x_2, inputs=x_1, grad_outputs=torch.ones_like(x_2), retain_graph=True)
loss.backward()

print("loss=", loss)
print("结束")

# J = torch.autograd.functional.jacobian(lambda z: linear(z), x_1)

# print("雅可比矩阵形状为：", J.shape)
# # 展开为二维矩阵
# J1 = J.reshape(x_2.numel(), x_1.numel())

# print("展开后的雅可比矩阵：", J1.shape)

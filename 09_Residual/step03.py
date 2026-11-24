"""
线性层雅可比矩阵测试
"""

import torch
import torch.nn as nn

torch.manual_seed(23)
x_1 = torch.randint(0, 10, (2, 3), dtype=torch.float32, requires_grad=True)
inter = 1 * x_1
inter.retain_grad()
linear = nn.Linear(3, 4, bias=False)
x_2 = linear(inter)
x_2.retain_grad()
print("x_2的形状为", x_2.shape)

# J = torch.autograd.functional.jacobian(lambda z: linear(z), x_1)

# print("雅可比矩阵形状为：", J.shape)
# # 展开为二维矩阵
# J1 = J.reshape(x_2.numel(), x_1.numel())

# print("展开后的雅可比矩阵：", J1.shape)


target = torch.randint(0, 10, (2, 4)).float()
criterion = nn.MSELoss()
loss = criterion(x_2, target)
# grad_loss_x2 = torch.autograd.grad(outputs=loss, inputs=x_2, retain_graph=True)[0]
# grad = torch.autograd.grad(outputs=x_2, inputs=inter, grad_outputs=grad_loss_x2, retain_graph=True)
grad = torch.autograd.grad(outputs=x_2.sum(), inputs=inter, retain_graph=True)
grad = torch.autograd.grad(outputs=x_2, inputs=inter, grad_outputs=torch.ones_like(x_2), retain_graph=True)
print(f"{grad=}")
inter.grad = None
x_2.grad = None

loss.backward()

print("结束")

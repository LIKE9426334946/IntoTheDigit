import torch
import torch.nn as nn

x = torch.randn(1, 3, 5, 5)

conv0 = nn.Conv2d(3, 3, 3, padding=1)
conv1 = nn.Conv2d(3, 3, 3, padding=1)

x_1 = conv0(x)
x_2 = conv1(x_1)

target = torch.randn_like(x_2)
loss = ((x_2 - target) ** 2).sum()

# 第一次反向传播
grad = torch.autograd.grad(
    x_2.sum(),
    x_1,
    retain_graph=False
)

# 第二次反向传播
loss.backward()
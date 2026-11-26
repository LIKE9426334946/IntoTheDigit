# 看看是如何传播梯度的


import torch
import torch.nn as nn

torch.manual_seed(23)

x = torch.randn((1, 3, 5, 5), requires_grad=True)
target = torch.randn((1, 3, 1, 1))

m = nn.AdaptiveAvgPool2d((1, 1))
y = m(x)
y.retain_grad()
loss = ((y - target) ** 2).sum()
loss.backward()


print("结束")


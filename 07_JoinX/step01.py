"""
x每次都会参与计算
先让每个通道共享一个参数
"""

import torch
import torch.nn as nn

torch.manual_seed(23)

x = torch.randn((1, 3, 5, 5))
target = torch.randint(0, 2, (1, 1, 5, 5)).float()

k1 = torch.randn((5, 5), requires_grad=True)
k2 = torch.randn((5, 5), requires_grad=True)
k3 = torch.randn((5, 5), requires_grad=True)

conv1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
conv3 = nn.Conv2d(3, 1, 3, padding=1, bias=False)

f1 = x * k1
f2 = x * k2
f3 = x * k3
f1.retain_grad()
f2.retain_grad()
f3.retain_grad()


c1 = conv1(f1) + f2
c2 = conv2(c1) + f3
c1.retain_grad()
c2.retain_grad()

criterion = nn.BCEWithLogitsLoss()


y = conv3(c2)
y.retain_grad()
loss = criterion(y, target)
loss.backward()

print(f"loss的值为{loss:.4f}")
print("执行结束")

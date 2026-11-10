"""
#  用来观察BCEWithLogitsLoss是如何传播梯度的
# 定义一层卷积网络

# BCEWithLogitsLoss对于四维输入，要求：
输入(N,C,H,W)
标签(N,1,H,W)

增加了迭代过程，看看loss是如何变化的

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(23)

x = torch.randint(0, 10, (1, 3, 5, 5), dtype=torch.float32)
target = torch.randint(0, 2, (1, 1, 3, 3)).float()

conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, bias=False)

criterion = nn.BCEWithLogitsLoss()

print(f"target:{target}")

for i in range(100):
    y = conv(x)
    y.retain_grad()
    print(f"y={y}")
    loss = criterion(y, target)
    loss.backward()
    with torch.no_grad():
        conv.weight -= 0.1 * conv.weight
    print(f"loss={loss}")
    
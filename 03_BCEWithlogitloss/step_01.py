"""
#  用来观察BCEWithLogitsLoss是如何传播梯度的
# 定义一层卷积网络

# CrossEntropyLoss对于四维输入，要求：
输入(N,C,H,W)
标签(N,1,H,W)


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

y = conv(x)
y.retain_grad()
print(y)
loss = criterion(y, target)
loss.backward()
print(loss)

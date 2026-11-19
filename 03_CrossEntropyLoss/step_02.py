"""
进行反向传播，看看loss的值会怎么样
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(23)

x = torch.randint(0, 10, (1, 3, 5, 5), dtype=torch.float32)
target = torch.randint(0, 3, (1, 3, 3))

conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False)

criterion = nn.CrossEntropyLoss()

print(f"target:{target}")


for i in range(30):
    conv.zero_grad()  # 清空梯度
    y = conv(x)
    loss = criterion(y, target)
    loss.backward()
    with torch.no_grad():
        conv.weight -= 0.01 * conv.weight.grad
    print(f"Epoch {i+1}, Loss: {loss.item():.6f}")

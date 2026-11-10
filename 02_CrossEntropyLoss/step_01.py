"""
#  用来观察CrossEntropyLoss是如何传播梯度的
# 定义一层卷积网络

# CrossEntropyLoss对于四维输入，要求：
输入(N,C,H,W)
标签(N,H,W)

也就是y的每个分量梯度对y的类别维度做softmax得到的概率，然后只有正确标签对应的哪个位置会减去1，最后所有分量再除以尺寸图的尺寸，得到的结果就是y的梯度了
一个很有意思的点，这三个类别的梯度相加一定等于0
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

y = conv(x)
y.retain_grad()
print(y)
loss = criterion(y, target)
loss.backward()
print(loss)

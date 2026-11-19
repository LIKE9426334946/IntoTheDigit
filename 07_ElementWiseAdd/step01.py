# 看看两个形状相同的通道相加是如何传递梯度
# 通道相加后用使用卷积进行融合

import torch
import torch.nn as nn

torch.manual_seed(23)

target = torch.randint(0, 2, (1, 1, 2, 2)).float()
x_A = torch.randn((1, 3, 5, 5))
x_B = torch.randn((1, 3, 5, 5))

conv_A = nn.Conv2d(3, 2, 3, bias=False)
conv_B = nn.Conv2d(3, 2, 3, bias=False)

feature_A = conv_A(x_A)
feature_A.retain_grad()
feature_B = conv_B(x_B)
feature_B.retain_grad()

# feature.shape=[1, 2, 3, 3]
# feature.dtype=torch.float32
feature = feature_A + feature_B
feature.retain_grad()

conv = nn.Conv2d(2, 1, 2, bias=False)
y = conv(feature)
y.retain_grad()

criterion = nn.BCEWithLogitsLoss()
loss = criterion(y, target)
loss.backward()

print("反向传播结束")

"""
手工调整梯度的例子
"""

import torch

torch.manual_seed(23)

x = torch.tensor([[3, 2], [5, 4], [9, 1]], dtype=torch.float32)
W = torch.nn.Parameter(torch.tensor([[-0.33, 2.0, 3.0], [4.0, 5.0, 6.0]]))

y = x @ W
t = torch.randint(0, 10, (3, 3))
# 构造一个简单的损失函数
loss = ((y - t) ** 2).sum()
print("loss的值为", loss)
# 反向传播
loss.backward()

print("参数 W：")
print(W.data)

print("梯度 W.grad：")

print(W.grad)

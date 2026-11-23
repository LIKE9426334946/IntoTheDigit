import torch
import torch.nn as nn

x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
W = nn.Linear(2, 2, bias=False)

y = W(x)
loss = y.sum()

# 计算一次梯度
grad = torch.autograd.grad(loss, x)

print(x.grad)
loss.backward()
print(loss)
print(grad)
print(x.grad)
print("结束")

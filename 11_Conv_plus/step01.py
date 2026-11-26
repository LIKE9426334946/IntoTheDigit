import torch
import torch.nn as nn

torch.manual_seed(23)

x = torch.randint(0, 10, (1, 3, 5, 5), dtype=torch.float32, requires_grad=True)
W = nn.Conv2d(3, 2, 2, bias=False)
y = W(x)
loss = y.sum()
loss.backward()

# 对x的梯度，是这样
J = torch.autograd.functional.jacobian(func=lambda x: W(x), inputs=x)

print(J.shape)
print(J[:, :, :, :, 0, 0, 1, 1])


print("结束")

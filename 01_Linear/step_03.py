import torch

torch.manual_seed(23)

x = torch.tensor([[3, 2], [5, 4], [9, 1]], dtype=torch.float32, requires_grad=True)
W = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

y = x @ W
y.retain_grad()
t = torch.randint(0, 10, (3, 3))
# 构造一个简单的损失函数
loss = ((y - t) ** 2).sum()

grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), retain_graph=True)

loss.backward()

import torch
import torch.nn as nn

torch.manual_seed(23)

x_1 = torch.randint(0, 10, (2, 2), dtype=torch.float32, requires_grad=True)
W = nn.Parameter(torch.randn(2, 2))
y = x_1 @ W
y.retain_grad()
target = torch.randint(0, 10, (2, 2), dtype=torch.float32)
loss = ((y - target) ** 2).sum()


grad_loss_y = torch.autograd.grad(outputs=loss, inputs=y, retain_graph=False)
grad_y_loss = torch.autograd.grad(outputs=y.sum(), inputs=x_1, retain_graph=False)
print(grad_y_loss)

print("结束")

"""
看看残差连接梯度是如何传播的，没有relu函数
"""

import torch
import torch.nn as nn

torch.manual_seed(23)

x = torch.randn((1, 3, 5, 5))
target = torch.randint(0, 3, (1, 5, 5))


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x_0):
        x_1 = self.conv_0(x_0)
        x_1.retain_grad()
        x_2 = self.conv_1(x_1)
        x_2.retain_grad()
        J = torch.autograd.functional.jacobian(lambda z: self.conv_1(z), x_1)
        out = x_1 + x_2
        out.retain_grad()
        return out, x_1, x_2, J


model = MyModel()

criterion = nn.CrossEntropyLoss()

logits, x_1, x_2, J = model(x)

loss = criterion(logits, target)
x_2 = x_2
# grad = torch.autograd.grad(outputs=x_2, inputs=x_1, grad_outputs=torch.ones_like(x_2), retain_graph=True)
grad = torch.autograd.grad(outputs=x_2, inputs=x_1, grad_outputs=torch.ones_like(x_2), retain_graph=True)
print(f"{x_1.grad=}")


loss.backward()
print(f"{x_1.grad=}")

"""
x每次都会参与计算
先让每个通道共享一个参数
把k1，k2，k3注册为可学习参数
"""

p = print
import torch
import torch.nn as nn

torch.manual_seed(23)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.k1 = nn.Parameter(torch.randn(5, 5))
        self.k2 = nn.Parameter(torch.randn(5, 5))
        self.k3 = nn.Parameter(torch.randn(5, 5))
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3, 1, 3, padding=1, bias=False)

    def forward(self, x):
        f1 = x * self.k1
        f2 = x * self.k2
        f3 = x * self.k3

        c1 = self.conv1(f1) + f2
        c2 = self.conv2(c1) + f3
        return self.conv3(c2)


x = torch.randn((1, 3, 5, 5))
target = torch.randint(0, 2, (1, 1, 5, 5)).float()

model = MyModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
p(model.k1)
p(model.k1.requires_grad)

# for i in range(50):
#     optimizer.zero_grad()
#     y = model(x)
#     loss = criterion(y, target)
#     loss.backward()
#     optimizer.step()
#     print(f"第{i+1}次loss的值为{loss:.4f}")

# with torch.no_grad():
#     torch.onnx.export(model, (x,), "demo/01.onnx", external_data=False)
# print("执行结束")

# 先进行一个参数的SGD计算
# 先研究momentum

import torch
import torch.nn as nn

torch.manual_seed(23)

x = torch.tensor([3.0])
target = torch.tensor([10])

model = nn.Linear(1, 1, bias=False)
nn.init.constant_(model.weight, 2.0)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for i in range(10):
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()

    print("一次参数更新结束")

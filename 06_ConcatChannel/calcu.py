import torch

# 计算step_01输出的损失函数
x = torch.tensor([-0.1336, 0.1266, 0.0967, 0.0632])
y = torch.sigmoid(x)
print(y)


# 求损失函数
loss = -torch.log(torch.tensor([0.4666, (1 - 0.5316), (1 - 0.5242), 0.5158]))
print(loss)

print(loss.sum() / 4)
# 得到损失函数0.7314

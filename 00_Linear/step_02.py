import torch

torch.manual_seed(23)

x = torch.tensor([[3, 2], [5, 4], [9, 1]], dtype=torch.float32)
W = torch.nn.Parameter(torch.tensor([[0.75, 1, 0.4], [0.5, 0.3, 0.5]]))
t = torch.randint(0, 10, (3, 3))

for i in range(100):
    W.grad = None
    y = x @ W

    # 构造一个简单的损失函数
    loss = ((y - t) ** 2).sum()
    print(f"第{i+1}次loss的值为", loss)
    # 反向传播
    loss.backward()
    with torch.no_grad():
        W -= 0.001 * W.grad


print("参数 W：")
print(W.data)

print("梯度 W.grad：")

print(W.grad)

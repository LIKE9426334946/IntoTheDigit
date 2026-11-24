import torch

x = torch.tensor([[1.0, 2], [3, 4]], requires_grad=True)
W = torch.tensor([[1.0, 2], [3, 4]], requires_grad=True)

y = x @ W
y.retain_grad()
loss = (y**2).sum()

# J = torch.autograd.functional.jacobian(
#     func=lambda x: x @ W,
#     inputs=x
# )
# print(J.shape)
# print(J)

loss.backward()
print(y.grad)

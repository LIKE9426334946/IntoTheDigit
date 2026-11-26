"""
线性层雅可比矩阵测试
"""

import torch
import torch.nn as nn

X_1 = torch.rand((2, 3))
X_2 = torch.rand((2, 3))
print(torch.allclose(X_1, X_2))


a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.000001, 2.000001, 3.000001])

print(torch.allclose(a, b))


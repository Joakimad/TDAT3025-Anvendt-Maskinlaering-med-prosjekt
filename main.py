import torch

# Tensor are basically arrays
x = torch.tensor([5, 3])
y = torch.tensor([2, 1])
print(x * y)

# Create a zero tensor in a given size
x = torch.zeros([2, 5])
print(x)

# Give you the size and shape of the tensor
print(x.shape)

# Gives a random tensor (0-1) in a given shape
y = torch.rand([2, 5])
print(y)

# Reshape tensor, requires declaration
y = y.view([1, 10])
print(y)

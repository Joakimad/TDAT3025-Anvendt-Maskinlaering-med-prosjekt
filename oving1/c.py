import torch
import matplotlib.pyplot as plt

# Find circumference based on days
# Observed/training input and output
days = []
circumference = []
with open("day_head_circumference.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        l1, l2 = line.strip().split(",")
        days.append(float(l1))
        circumference.append(float(l2))

x_train = torch.tensor(days).reshape(-1, 1)
y_train = torch.tensor(circumference).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        # f (x) = 20σ(xW + b) + 31
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Loss function
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
learning_rate = 1e-6
epochs = 50000
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$y = f(x) = 20σ(xW + b) + 31')
plt.legend()
plt.show()

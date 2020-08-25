import torch
import matplotlib.pyplot as plt
import pandas as pd

# Variables we tweak to find the lowest loss
learning_rate = 1e-6
epochs = 80000

# Find circumference based on days
# Observed/training input and output
data = pd.read_csv("day_head_circumference.csv")

x_data = data["day"].tolist()
y_data = data["head circumference"].tolist()

x_train = torch.tensor(x_data, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(y_data, dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        # f(x) = 20σ(xW + b) + 31
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Loss function
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')

# Labels
plt.xlabel('x')
plt.ylabel('y')

# Have to sort x_train to plot correctly
x, indices = torch.sort(x_train, 0)
plt.plot(x, model.f(x).detach(), label='$y = f(x) = 20*σ(xW+b)+31$', c="red")

plt.legend()
plt.show()

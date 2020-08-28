import torch
import matplotlib.pyplot as plt
import pandas as pd

# Variables we tweak to find the lowest loss
learning_rate = 1e-5
epochs = 80000

# Find age in days based on length and weight.
# Observed/training input and output
data = pd.read_csv("day_length_weight.csv")

x_data = [data["length"].tolist(), data["weight"].tolist()]
y_data = data["day"].tolist()

x_train = torch.tensor(x_data, dtype=torch.float).t()
y_train = torch.tensor(y_data, dtype=torch.float).t().reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # requires_grad enables calculation of gradients.
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
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
fig = plt.figure().gca(projection='3d')

# Data points
fig.scatter(data["length"].tolist(), data["weight"].tolist(), data["day"].tolist(), c='orange')

# Regression line
fig.scatter(data["length"].tolist(), data["weight"].tolist(), model.f(x_train).detach(), label='$y = f(x) = xW+b$')

# Labels
fig.set_xlabel('Length')
fig.set_ylabel('Weight')
fig.set_zlabel('Days')

plt.show()

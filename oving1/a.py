import torch
import matplotlib.pyplot as plt
import pandas as pd

# Variables we tweak to find the lowest loss
learning_rate = 0.0001
epochs = 155000

# Find weight based on length
# Observed/training input and output
data = pd.read_csv("length_weight.csv")

x_data = data["length"].tolist()
y_data = data["weight"].tolist()

x_train = torch.tensor(x_data, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(y_data, dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    # torch.nn.functional.mse_loss(self.f(x), y)
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

# Plot
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')

plt.legend()
plt.show()

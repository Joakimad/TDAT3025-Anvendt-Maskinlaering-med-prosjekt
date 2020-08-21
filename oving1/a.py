import torch
import matplotlib.pyplot as plt

# Find weight based on length
# Observed/training input and output
lengths = []
weights = []
with open("length_weight.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        l1, l2 = line.strip().split(",")
        lengths.append(float(l1))
        weights.append(float(l2))

x_train = torch.tensor(lengths).reshape(-1, 1)
y_train = torch.tensor(weights).reshape(-1, 1)


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
learning_rate = 0.0001
epochs = 155000
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
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
plt.legend()
plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np

# Find age in days based on length and weight.
# Observed/training input and output
days = []
weight_length = []
with open("day_length_weight.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        l1, l2, l3 = line.strip().split(",")
        days.append(float(l1))
        weight_length.append([float(l2), float(l3)])

x_train = torch.tensor(weight_length)
y_train = torch.tensor(days)


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
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
learning_rate = 1e-6
epochs = 1500
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# VISUALIZATION
# Set title for window
fig = plt.figure('Linear regression: 3D')

# Set type of projection
plot1 = fig.add_subplot(111, projection='3d')

plot1_info = fig.text(0.01, 0.02, '')
plot1_info.set_text(
    '$W=\\left[\\stackrel{%.2f}{%.2f}\\right]$\n$b=[%.2f]$\n$loss = \\frac{1}{n}\\sum_{i=1}^{n}(f(\\hat x^{(i)}) - \\hat y^{(i)})^2 = %.2f$' %
    (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))

# Plot data points
plot1.plot(x_train[:, 0].squeeze(),
           x_train[:, 1].squeeze(),
           y_train.squeeze(),
           'o',
           label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$',
           color='blue')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color='green', label='$y = f(x) = xW+b$')

plot1_loss = []
for i in range(0, x_train.shape[0]):
    line, = plot1.plot([0, 0], [0, 0], [0, 0], color='red')
    plot1_loss.append(line)
    if i == 0:
        line.set_label('$|f(\\hat x^{(i)})-\\hat y^{(i)}|$')

plot1.set_xlabel('$x_1$')
plot1.set_ylabel('$x_2$')
plot1.set_zlabel('$y$')
plot1.legend(loc='upper left')
plot1.set_xticks([])
plot1.set_yticks([])
plot1.set_zticks([])
plot1.w_xaxis.line.set_lw(0)
plot1.w_yaxis.line.set_lw(0)
plot1.w_zaxis.line.set_lw(0)

plt.show()

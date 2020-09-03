import torch
import matplotlib.pyplot as plt
import numpy as np

# Variables we tweak to find the lowest loss
learning_rate = 0.1
epochs = 10000

# Observed/training input and output
x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_train = torch.tensor([[1.], [1.], [1.], [0.]])


class SigmoidModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((2, 1), dtype=torch.float, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float, requires_grad=True)

    # logits er modellprediktoren før normalisering ved hjelp av σ eller softmax
    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Loss function
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure("NAND Operator")
plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                               label="$y=f(x)=\\sigma(xW+b)$")

# Data points
plot1.plot(x_train[:, 0].squeeze(),
           x_train[:, 1].squeeze(),
           y_train[:, 0].squeeze(),
           'o',
           label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
           color="blue")

plot1_info = fig.text(0.01, 0.02, "$W=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\n$b=[%.2f]$\n$loss = -\\frac{1}{n}\\sum_{"
                                  "i=1}^{n}\\left [ \\hat y^{(i)} "
                                  "\\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$ "
                      % (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))

# Grids
x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10)),
                                dtype=torch.float)
y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.float)
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

# Table
table = plt.table(cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                  cellLoc="center",
                  loc="upper right")
table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[0., 0.]], dtype=torch.float)).detach())
table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[0., 1.]], dtype=torch.float)).detach())
table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[1., 0.]], dtype=torch.float)).detach())
table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[1., 1.]], dtype=torch.float)).detach())

plt.savefig('NAND')
plt.show()

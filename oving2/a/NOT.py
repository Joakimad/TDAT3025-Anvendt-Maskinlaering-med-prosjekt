import torch
import matplotlib.pyplot as plt
import numpy as np

# Variables we tweak to find the lowest loss
learning_rate = 0.1
epochs = 10000

# Observed/training input and output
x_train = torch.tensor([[0.], [1.]])
y_train = torch.tensor([[1.], [0.]])


class SigmoidModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

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
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')

# Labels
plt.xlabel('x')
plt.ylabel('y')

# Plot
out = torch.reshape(torch.tensor(np.linspace(0, 1, 100).tolist()), (-1, 1))
x, indices = torch.sort(out, 0)
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')

plt.legend()
plt.savefig('NOT')
plt.show()

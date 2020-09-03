import torch
import torchvision
import matplotlib.pyplot as plt

# Variables
learning_rate = 0.01
epochs = 0

# Training set
mnist_train = torchvision.datasets.MNIST('d/data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 28 * 28).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

# Test set
mnist_test = torchvision.datasets.MNIST('d/data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 28 * 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1


class SoftmaxModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((28 * 28, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)

    # Logits
    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = SoftmaxModel()

optimizer = torch.optim.SGD([model.W, model.b], learning_rate, momentum=0.5)

acc = -1
loss = -1
while acc < 0.9:
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()
    optimizer.zero_grad()

    acc = model.accuracy(x_test, y_test).item()
    loss = model.loss(x_test, y_test).item()
    print("Epoch: %s, Loss: %s, Accuracy: %s" % (epochs + 1, loss, acc))
    epochs += 1

print("Loss: %s, Accuracy: %s" % (loss, acc))

# Show the input of the first observation in the training set
plt.imshow(x_train[0, :].reshape(28, 28))
plt.show()

# Print the classification of the first observation in the training set
print(y_train[0, :])

# Save pictures of W after optimization
for i in range(10):
    plt.imsave("d/%i.png" % (i+1), model.W[:, i].reshape(28, 28).detach())



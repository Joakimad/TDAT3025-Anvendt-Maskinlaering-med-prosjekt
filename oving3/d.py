import torch
import torch.nn as nn
import torchvision

# Variables
epochs = 20
learning_rate = 0.001

train_data = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = train_data.data.reshape(-1, 1, 28, 28).float()
y_train = torch.zeros((train_data.targets.shape[0], 10))
y_train[torch.arange(train_data.targets.shape[0]), train_data.targets] = 1

test_data = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = test_data.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((test_data.targets.shape[0], 10))
y_test[torch.arange(test_data.targets.shape[0]), test_data.targets] = 1

# Normalize
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Batches
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()

        self.logits = nn.Sequential(
            # First layer
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second layer
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 100),
            nn.Linear(100, 10)
        )

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalModel()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
for epoch in range(epochs):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("%s: accuracy = %s" % (epoch, model.accuracy(x_test, y_test)))

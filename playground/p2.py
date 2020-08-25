import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Gets datasets and molds them to Tensor form.
# MNIST is hand drawn numbers dataset. 28x28 image. Numbers from 0-9.
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Batch size = how many are passed to the model at the time.
# Usually uses base8 numbers. No particular reason.
# Bigger number decreases training time. You don't always want biggest, find the sweet spot.
# It is better to send data in batches as it will improve the computer's optimization.
# Shuffle. Used to improve the computer's optimization. Computer will want to take easiest way.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

# y is the first number, x is the image data.
x, y = data[0][0], data[1][0]
print(y)
print(x.shape)

# Pytorch has a weird shape with adding a 1 at the begninning. We have to reshape it to a 28x28 image.
plt.imshow(data[0][0].view([28, 28]))
plt.show()

total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i] / total * 100}")

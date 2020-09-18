import torch
import torch.nn as nn

# Variables
lr = 0.001
epochs = 750
test_words = ['hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son ',
              'ht  ', 'rt  ', 'ct  ', 'flt ', 'mtt ', 'cp  ', 'sn  ',
              'hats', 'rats', 'cats', 'caps', 'sons']


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_enc = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # h
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # a
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # t
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # r
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # c
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # f
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # l
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # m
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # p
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # s
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # o
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # n
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 🎩
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 🐀
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 🐈
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 🏢
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 👨
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 🧢
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 👦
]
encoding_size = len(char_enc)

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n',
                 '🎩', '🐀', '🐈', '🏢', '👨', '🧢', '👦']

# Words are written in columns
x_train = torch.tensor([
    [char_enc[1], char_enc[4], char_enc[5], char_enc[6], char_enc[8], char_enc[5], char_enc[10]],
    [char_enc[2], char_enc[2], char_enc[2], char_enc[7], char_enc[2], char_enc[2], char_enc[11]],
    [char_enc[3], char_enc[3], char_enc[3], char_enc[2], char_enc[3], char_enc[9], char_enc[12]],
    [char_enc[0], char_enc[0], char_enc[0], char_enc[3], char_enc[3], char_enc[0], char_enc[0]]])
#       hat          rat          cat         flat          matt          cap            son

y_train = torch.tensor([char_enc[13],   # 🎩
                        char_enc[14],   # 🐀
                        char_enc[15],   # 🐈
                        char_enc[16],   # 🏢
                        char_enc[17],   # 👨
                        char_enc[18],   # 🧢
                        char_enc[19]])  # 👦


# Takes the given text and outputs an emoji
def to_emoji(text):
    model.reset(1)
    letters = []
    for char in text:
        temp = [char_enc[index_to_char.index(char)]]
        letters.append(temp)
    letters.append(temp)
    letters = torch.tensor(letters)
    y = model.f(letters)
    return index_to_char[y.argmax(1)]


model = LongShortTermMemoryModel(encoding_size)
optimizer = torch.optim.RMSprop(model.parameters(), lr)

for epoch in range(epochs):
    model.reset(x_train.size(1))
    loss = model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 50 == 0:
        print("Epoch: %s" % epoch)
        for word in test_words:
            print("%s - %s" % (word, to_emoji(word)))
        print()

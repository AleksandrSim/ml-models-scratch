import torch.nn as nn
import torch
import torch.nn.functional as F
from global_variables import *

# FOR Q1,4
class SimpleFeedForward(nn.Module):
    def __init__(self, size_in, size_out, device, lr, bias):
        super(SimpleFeedForward, self).__init__()
        self.linearStack = nn.Sequential(

            nn.Linear(size_in, insurability_nodes_per_layer, bias=bias),
            nn.Sigmoid(),
            nn.Linear(insurability_nodes_per_layer, size_out, bias=bias)
        )
        self.learningRate = lr
        self.lossFunction = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr, momentum=0.9)

    def forward(self, X):
        X = self.linearStack(X)
        return X

    def softmax(self, tnes):
        """Softmax function to normalize the output."""
        return torch.exp(tnes) / torch.sum(torch.exp(tnes))


# FOR Q2-3
# Here we declare the forward feed class for our NN
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_classes):
        super(FeedForward, self).__init__()
        # Defining our Layers here
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, 288)
        self.l2 = nn.Linear(288, 200)
        self.l3 = nn.Linear(200, 150)
        self.l4 = nn.Linear(150, 100)
        self.l5 = nn.Linear(100, 50)
        self.l6 = nn.Linear(50, 25)
        self.l7 = nn.Linear(25, num_of_classes)

    def forward(self, x):
        out = self.leaky_relu(self.l1(x))
        out = self.leaky_relu(self.l2(out))
        out = self.leaky_relu(self.l3(out))
        out = self.leaky_relu(self.l4(out))
        out = self.leaky_relu(self.l5(out))
        out = self.leaky_relu(self.l6(out))
        out = self.l7(out)
        return out

    def compute_l2_loss(self, w):
        return torch.square(w).sum()


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)

        return x

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

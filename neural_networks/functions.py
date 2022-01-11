from math import exp

class Layer_Function():
    def __init__(self):
        self.name = "blank"

    def forward(self, x, weight):
        pass

    def backward(self, x):
        pass

    def __str__(self):
        return self.name

class Loss_Function():
    def __init__(self):
        self.name = "loss"

    def forward(self, pred, actual):
        pass

    def backward(self):
        pass

    def __str__(self):
        return self.name

class Linear(Layer_Function):
    def __init__(self):
        super(Linear, self).__init__()
        self.name = "linear"

    def forward(self, x, weight):
        return x * weight

    def backward(self, x):
        return 1


class Sigmoid(Layer_Function):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = "sigmoid"

    def forward(self, x, weight):
        return 1 / (1 - exp(x))

    def backward(self, x):
        return 1 - 1 / (1 - exp(x))

class CE_Loss(Loss_Function):
    def forward(self, pred, actual):
        #calculate cross entropy loss
        pass

    def backward(self):
        #calc derivative of cross entropy loss
        pass



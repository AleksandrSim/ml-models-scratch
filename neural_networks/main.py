import numpy as np

from simple_two_hidden_layers import one_hot_code_encoding, read_data, DeepNeuralNetwork


if __name__=='__main__':
    data = read_data('train.csv')
    data_val = read_data('valid.csv')
    y = np.array([i[0] for i in data])
    X = (np.array([i[1] for i in data]) / 255).astype('float32')
    y_valid = np.array([i[0] for i in data_val])
    X_valid = (np.array([i[1] for i in data_val]) / 255).astype('float32')
    y = np.array([one_hot_code_encoding(i) for i in y])
    y_valid = np.array([one_hot_code_encoding(i) for i in y_valid])
    net = DeepNeuralNetwork(sizes=[784,128,64,10])
    net.train(X, y, X_valid, y_valid)
















from simple_feed_forward import train_insurability, validate_insurability
from train_minst import *
from feed_forward_from_scratch import *
from data_tensor_conversions import *
from insurability_processing import *

def read_mnist(file_name):
    data_set = []
    with open(file_name, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show_mnist(file_name, mode):
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == "pixels":
                if data_set[obs][1][idx] == "0":
                    print(" ", end="")
                else:
                    print("*", end="")
            else:
                print("%4s " % data_set[obs][1][idx], end="")
            if (idx % 28) == 27:
                print(" ")
        print("LABEL: %s" % data_set[obs][0], end="")
        print(" ")


def classify_insurability():
    train = InsurabilityDataset("three_train.csv")
    valid = InsurabilityDataset("three_valid.csv")
    test = InsurabilityDataset("three_test.csv")
    train_insurability(train, test, valid)


def classify_mnist():
    train = read_mnist("mnist_train.csv")
    valid = read_mnist("mnist_valid.csv")
    test = read_mnist("mnist_test.csv")
    # show_mnist("mnist_test.csv", "pixels")
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    # use the 4th argument to apply the regularizer in this case we dont
    # the 5th argument if true will apply dimensionality reduction
    # use true for the last argument if you want to use a CNN model and False for simple NN
    train_minst_models(train, test, valid, regulate=True, use_dim_reduction=True, use_cnn=True)


def classify_mnist_reg():
    train = read_mnist("mnist_train.csv")
    valid = read_mnist("mnist_valid.csv")
    test = read_mnist("mnist_test.csv")
    # show_mnist("mnist_test.csv", "pixels")
    # add a regularizer of your choice to classify_mnist()
    # (a FFNN is fine) and produce evaluation metrics
    # use the 4th argument to apply the regularize in this case we do
    # the 5th argument if true will apply dimensionality reduction
    # use true for the last argument if you want to use a CNN model and False for simple NN
    train_minst_models(train, test, valid, regulate=False, use_dim_reduction=True, use_cnn=True)


def classify_insurability_manual():
    train = read_insurability("three_train.csv")
    valid = read_insurability("three_valid.csv")
    test = read_insurability("three_test.csv")
    train_model_from_scratch(train, valid, test)


def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()


if __name__ == "__main__":
    main()

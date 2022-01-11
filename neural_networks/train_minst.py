import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from dimensionality import *
from confusion_matrix import *
from global_variables import *
from nn_models import *
from data_tensor_conversions import *


def train_minst_models(train, test, valid, regulate, use_dim_reduction, use_cnn):
    # Get the dimensions of the training data and store them in a variable
    # Perform dimensionality reduction, but first convert every element to int
    if use_dim_reduction:
        list_of_data = [train, test, valid]
        list_of_data = dimensionality_reduction(list_of_data)
        train = list_of_data[0]
        test = list_of_data[1]
        valid = list_of_data[2]
    # Get global variables and input size
    input_size = get_minst_input_size(train)
    # Assign the forward feed class to a model variable
    if use_cnn:
        if not use_dim_reduction:
            model = Net(500)
        else:
            model = Net(320)
    else:
        model = FeedForward(input_size, mnist_hidden_size, mnist_num_of_classes)
    # Define a loss function, in this case we choose Cross Entropy Loss
    loss_function = nn.CrossEntropyLoss()
    # Declaring an optimization function (in this case we choose Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=mnist_learning_rate, weight_decay=mnist_weight_decay)

    '''CONVERTING THE INPUT DATA TO TENSORS'''
    X = convert_data_to_tensor(train, input_size)
    if use_cnn:
        if not use_dim_reduction:
            X = torch.tensor(np.array(X).reshape(2000, 1, 28, 28))
        else:
            zeroes = torch.zeros(2000, 29)
            X = torch.cat((X, zeroes), dim=-1)
            X = torch.tensor(np.array(X).reshape(2000, 1, 24, 24))

    # Classes Definition
    y = torch.tensor(np.array([int(i[0]) for i in train]))
    classes = []

    for i in y:
        if i not in classes:
            classes.append(int(i))

    # use TensorDataset for our features and classes (x,y)
    my_dataset = TensorDataset(X, y)
    my_dataloader = DataLoader(my_dataset)

    val_X = convert_data_to_tensor(valid, input_size)
    val_Y = torch.tensor(np.array([int(i[0]) for i in valid]))
    if use_cnn:
        if not use_dim_reduction:
            val_X = torch.tensor(np.array(val_X).reshape(200, 1, 28, 28))
        else:
            val_X = torch.cat((val_X, torch.zeros(200, 29)), dim=1)
            val_X = torch.tensor(np.array(val_X).reshape(200, 1, 24, 24))

    valid_dataloader = TensorDataset(val_X, val_Y)
    valid = DataLoader(valid_dataloader)

    '''Training the NN'''
    list_to_plot = []
    val_to_plot = []
    for epoch in range(mnist_num_of_epochs):
        all = 0
        correct = 0
        val_pred = model(val_X)
        _, val_pred = torch.max(val_pred, 1)
        all += val_Y.size(0)
        correct += (val_pred == val_Y).sum().item()
        accuracy = correct / all
        val_to_plot.append(accuracy)
        print(accuracy)

        if accuracy > 0.95:
            break
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(my_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if regulate:
                l2_weight = 0.1
                l2_parameters = []
                for parameter in model.parameters():
                    l2_parameters.append(parameter.view(-1))
                l2 = l2_weight * model.compute_l2_loss(torch.cat(l2_parameters))
                loss = loss + l2
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                list_to_plot.append(running_loss / 2000)  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('\nFinished Training\n')

    # Plotting a graph
    y_axis = [i for i in range(len(list_to_plot))]
    plt.plot(y_axis, list_to_plot)
    plt.show()
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(list_to_plot)
    ax_right.plot(val_to_plot, color='red')
    plt.show()

    # Now it is time to test and get the accuracy results
    X = convert_data_to_tensor(test, input_size)

    if use_cnn:
        if not use_dim_reduction:
            X = torch.tensor(np.array(X).reshape(200, 1, 28, 28))
        else:
            print(X)
            X = torch.cat((X, torch.zeros(200, 29)), dim=1)
            X = torch.tensor(np.array(X).reshape(200, 1, 24, 24))

    y = torch.tensor(np.array([int(i[0]) for i in test]))
    # use TensorDataset for our features and classes (x,y)
    my_dataset = TensorDataset(X, y)
    test_loader = DataLoader(my_dataset)

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        pr_l = []
        act_l = []
        for i in test_loader:
            X, y = i

            predict = model(X)
            _, predict = torch.max(predict.data, 1)
            total += y.size(0)
            # print(y)
            correct += (predict == y).sum().item()
            for label, output in zip(y, predict):
                if label == output:
                    correct_pred[classes[label]] += 1
                pr_l.append(classes[output])
                total_pred[classes[label]] += 1
                act_l.append(classes[label])

    print("Total -> " + str(total))
    print("Correct -> " + str(correct))
    print("Accuracy -> " + str((correct / total) * 100) + " %")
    print("Correct Predictions -> ", correct_pred)
    print("Total Predictions -> ", total_pred)
    # calling the pretty print confusion matrix function
    print(pretty_print(confusion_matrix(act_l, pr_l)))

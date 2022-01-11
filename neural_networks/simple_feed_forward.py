from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from global_variables import *
from nn_models import *
from confusion_matrix import *


def train_insurability(train, test, valid):
    list_to_plot = []
    valid_to_plot = []
    model = SimpleFeedForward(3, 3, device, insurability_learning_rate, bias=True)
    train_loader = DataLoader(train, batch_size=insurability_batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=insurability_batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=insurability_batch_size, shuffle=False)

    for epoch in range(insurability_epochs):

        running_loss = 0.0
        accuracy = validate_insurability(valid_loader, model)
        valid_to_plot.append(accuracy)
        if accuracy >= 0.9:
            break

        for batch, (X, y) in enumerate(train_loader):
            pred = model(X)
            pred = model.softmax(pred)
            loss = model.lossFunction(pred, y)
            #model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch % 2000 == 1999:
                list_to_plot.append(running_loss / 2000)  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch + 1, running_loss / 2000))
                running_loss = 0.0

    plot_acc(list_to_plot, valid_to_plot)

    # Finally we test the results
    actual, predicted, correct_pred_count, total_pred_count, f1 = test_insurability(test_loader, model)
    print(pretty_print(confusion_matrix(actual, predicted)))
    print("Total -> " + str(total_pred_count))
    print("Correct -> " + str(correct_pred_count))
    print("Accuracy -> " + str((correct_pred_count / total_pred_count) * 100) + " %")
    print("F1_good Score -> " + str(f1[0]))
    print("F1_neutral Score -> " + str(f1[1]))
    print("F1_bad Score -> " + str(f1[2]))
    # calling the pretty print confusion matrix function



def validate_insurability(dataloader, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            _, pred = torch.max(pred.data, 1)
            _, y = torch.max(y.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return correct / total


# def test_insurability(dataloader, model):
#     total_predictions = 0
#     correct_predictions = 0
#     true_pos = 0
#     false_pos = 0
#     false_neg = 0
#     with torch.no_grad():
#
#         predictions = []
#         actual_labels = []
#
#         for batch, (X, y) in enumerate(dataloader):
#             pred = model(X)
#             _, pred = torch.max(pred.data, 1)
#             _, y = torch.max(y.data, 1)
#             correct_predictions += (pred == y).sum().item()
#             total_predictions += y.size(0)
#             predictions.append(pred.item())
#             actual_labels.append(y.item())
#
#         for i in range(len(actual_labels)):
#             if predictions[i] == 0 and actual_labels[i] == 0:
#                 true_pos = true_pos + 1
#             elif predictions[i] == 0 and (actual_labels[i] == 2 or actual_labels[i] == 1):
#                 false_pos = false_pos + 1
#             elif predictions[i] == 2 and (actual_labels[i] == 0 or actual_labels[i] == 1):
#                 false_neg = false_neg + 1
#
#         f1 = true_pos / (true_pos + 0.5 * (false_pos + false_neg))
#
#     return actual_labels, predictions, correct_predictions, total_predictions, f1

def test_insurability(dataloader, model):
    total_predictions = 0
    correct_predictions = 0
    good_true_pos = 0
    good_false_pos = 0
    good_false_neg = 0
    neutral_true_pos = 0
    neutral_false_pos = 0
    neutral_false_neg = 0
    bad_true_pos = 0
    bad_false_pos = 0
    bad_false_neg = 0

    with torch.no_grad():

        predictions = []
        actual_labels = []

        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            _, pred = torch.max(pred.data, 1)
            _, y = torch.max(y.data, 1)
            correct_predictions += (pred == y).sum().item()
            total_predictions += y.size(0)
            predictions.append(pred.item())
            actual_labels.append(y.item())

        for i in range(len(predictions)):
            print(predictions[i], actual_labels[i])


        for i in range(len(actual_labels)):
            # Good
            if predictions[i] == 0 and actual_labels[i] == 0:
                good_true_pos = good_true_pos + 1
            elif predictions[i] == 0 and (actual_labels[i] == 2 or actual_labels[i] == 1):
                good_false_pos = good_false_pos + 1
            elif (predictions[i] == 2 or predictions[i] == 1) and actual_labels[i] == 0:
                good_false_neg = good_false_neg + 1

            # Neutral
            if predictions[i] == 1 and actual_labels[i] == 1:
                neutral_true_pos = neutral_true_pos + 1
            elif predictions[i] == 1 and (actual_labels[i] == 0 or actual_labels[i] == 2):
                neutral_false_pos = neutral_false_pos + 1
            elif (predictions[i] == 0 or predictions[i] == 2) and actual_labels[i] == 1:
                neutral_false_neg = neutral_false_neg + 1

            # Bad
            if predictions[i] == 2 and actual_labels[i] == 2:
                bad_true_pos = bad_true_pos + 1
            elif predictions[i] == 2 and (actual_labels[i] == 0 or actual_labels[i] == 1):
                bad_false_pos = bad_false_pos + 1
            elif (predictions[i] == 0 or predictions[i] == 1) and actual_labels[i] == 2:
                bad_false_neg = bad_false_neg + 1

        f1_good = good_true_pos / (good_true_pos + 0.5 * (good_false_pos + good_false_neg))
        f1_neutral = neutral_true_pos / (neutral_true_pos + 0.5 * (neutral_false_pos + neutral_false_neg))
        f1_bad = bad_true_pos / (bad_true_pos + 0.5 * (bad_false_pos + bad_false_neg))

        f1 = [f1_good, f1_neutral, f1_bad]

    return actual_labels, predictions, correct_predictions, total_predictions, f1


def plot_acc(train_loss, valid_accuracies):
    # Plotting a graph
    y_axis = [i for i in range(len(train_loss))]
    plt.plot(y_axis, train_loss)
    plt.show()
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(train_loss)
    ax_right.plot(valid_accuracies, color='red')
    plt.show()

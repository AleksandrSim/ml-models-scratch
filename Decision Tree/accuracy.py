import math
import random
import ID3


def runningAverage(list_of_values):
    new_list = []
    for i in range(len(list_of_values)):
        # print(i)
        if i == 0:
            new_list.append(list_of_values[i])
        else:
            new_list.append(list_of_values[i] + new_list[i - 1])
    for i in range(len(list_of_values)):
        new_list[i] /= (i + 1)
    return new_list


def accuracy(data):
    withPruning = []
    withoutPruning = []
    # Shuffle 100 time, and split the data.
    for i in range(100):
        print("Iteration Step --> ", i)
        random.shuffle(data)
        # Spliting data.
        train_data = data[: len(data) // 2]
        valid_data = data[len(data) // 2: 3 * len(data) // 4]
        test_data = data[3 * len(data) // 4:]

        tree = ID3.ID3(train_data, 0)
        acc = ID3.test(tree, train_data)
        print("training accuracy: ", acc)
        acc = ID3.test(tree, valid_data)
        print("validation accuracy: ", acc)
        acc = ID3.test(tree, test_data)
        print("test accuracy: ", acc)

        ID3.prune(tree, valid_data)
        acc = ID3.test(tree, test_data)
        print("pruned tree train accuracy: ", acc)
        acc = ID3.test(tree, valid_data)
        print("pruned tree validation accuracy: ", acc)
        acc = ID3.test(tree, test_data)
        print("pruned tree test accuracy: ", acc)
        withPruning.append(acc)
        tree = ID3.ID3(train_data + valid_data, 0)
        acc = ID3.test(tree, test_data)
        print("no pruning test accuracy: ", acc)
        withoutPruning.append(acc)
    print(withPruning)
    print(withoutPruning)
    print("average with pruning", sum(withPruning) / len(withPruning), " without: ",
          sum(withoutPruning) / len(withoutPruning))

    return withPruning, withoutPruning

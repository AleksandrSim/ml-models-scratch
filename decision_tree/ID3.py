import accuracy
import data_cleaning
import entropy_ig
import global_variables
import graph
import list_dictionary_functions
import probability
from node import Node
from parse import parse
import operator
import unit_tests


def printDepthNode(N, depth):
    # for debugging, lets have it break after 2 layers of recursion
    if depth == 0:
        print("\n")
    print("\t" * depth, "Depth -> ", depth)
    print("\t" * depth, "Label -> ", N.label)
    print("\t" * depth, "Value -> ", N.value)
    if N.entropy:
        print("\t" * depth, "Entropy -> ", N.entropy)
    if N.information_gain:
        print("\t" * depth, "information_gain -> ", N.information_gain)
    if N.probability:
        print("\t" * depth, "probability -> ", N.probability)

    for key, value in N.children.items():
        print("\n")
        print("\t" * depth, "Child of ", N.label, " -> ", key)
        printDepthNode(value, depth + 1)


def ID3(examples, default):
    uniqueValues = list_dictionary_functions.dictionaryUniqueValues(examples, global_variables.target_attribute)

    if len(uniqueValues) == 1:
        node = Node()
        node.label = 'leaf'
        node.value = uniqueValues[0]
        global_variables.children += 1
        return node
    else:
        current_node = Node()
        global_variables.children += 1
        current_node.label, current_node.information_gain = entropy_ig.calculateMaxInformationGain(examples)
        current_node.probability = probability.probabilityForPrediction(examples, current_node.label)
        current_node.entropy = entropy_ig.entropyCalculator(examples, current_node.label)
        attributes = list_dictionary_functions.dictionaryUniqueValues(examples, current_node.label)

        # print(attributes)
        # print(current_node.label, examples)
        if len(attributes) < 2:
            current_node.label = 'leaf'
            current_node.value = default
        elif len(examples[0]) < 3:
            current_node.label = "leaf"
            current_node.value = default
        elif current_node.information_gain > global_variables.epsilon:
            for attribute in attributes:
                updatedExamples = list_dictionary_functions.removeAttributeFromList(examples, current_node.label,
                                                                                    attribute)

                updatedExamples = list_dictionary_functions.removeColumnFromDict(updatedExamples, current_node.label)
                # print(updatedExamples)
                current_node.children[attribute] = ID3(updatedExamples, default)

        else:
            current_node.label = 'leaf'
            current_node.value = default

        return current_node


# New pruning function:
def prune(node, examples):
    """
    Takes in a trained tree and a validation set of examples. Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    """
    initial_node = node
    first_stack = [initial_node]
    max_depth = 0

    while first_stack:
        initial_node = first_stack.pop()
        max_depth += 1
        # Will simply run through the tree and delete any nodes whose children are all leaf nodes
        for child in initial_node.children:
            first_stack.append(initial_node.children[child])

    accuracies = []
    trees = []
    current_threshold = 0
    max_threshold = 80

    while current_threshold <= max_threshold:
        current_threshold += 10
        temp_node = node
        stack = [temp_node]
        depth = 0

        while stack:
            temp_node = stack.pop()
            depth += 1

            if depth < (current_threshold * max_depth) / 100:
                # Will simply run through the tree and delete any nodes whose children are all leaf nodes
                for child in temp_node.children:
                    stack.append(temp_node.children[child])

        trees.append(temp_node)
        current_accuracy = test(temp_node, examples)
        accuracies.append(current_accuracy)

    print(accuracies)
    max_value = max(accuracies)
    max_index = accuracies.index(max_value)
    max_tree = trees[max_index]

    node = max_tree

    return node


def test(node, examples):
    """
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    """
    numeratedData = {}

    for i, k in enumerate(examples):
        numeratedData[i] = k

    actualTarget = []
    # print(numeratedData)
    for example in numeratedData:
        actualTarget.append(numeratedData[example]['Class'])

    predData = {}

    # for example in numeratedData:
    #   del predData[example]['Class']

    for i in numeratedData:
        predData[i] = {}
        for feature in numeratedData[i]:
            if feature != 'Class':
                predData[i][feature] = numeratedData[i][feature]

    rawPredictions = []

    for example in predData:
        prediction = evaluate(node, predData[example])
        rawPredictions.append(prediction)

    all = len(actualTarget)
    correct = 0

    for i in range(len(actualTarget)):
        if actualTarget[i] == rawPredictions[i]:
            correct += 1
    # uncomment this line for debugging purposes
    # print(actualTarget, rawPredictions)

    return correct / all


def evaluate(node, example):
    # Takes in a tree and one example.  Returns the Class value that the tree
    # assigns to the example.
    stack = [node]

    while stack:
        node = stack.pop()
        # Will simply run through the tree and delete any nodes whose children are all leaf nodes
        if node.label in example and node.children[example[node.label]]:
            # changing entropy here
            stack.append(node.children[example[node.label]])

    # node that is the last child whose key is example[node.labelexamp node.value
    # if node.probability:
    #             # The value of the example is the outer key of the dictionary:
    #     if node.probability.get(node.value):
    #         sub_dict = node.probability.get(node.value)
    #         print(sub_dict)
    #         max_value = max(sub_dict, key=sub_dict.get)
    #         return max_value
    #     else:
    #         return node.value
    # else:
    #     return node.value
    return node.value


def main():
    # 1) Import parse and use the parse function to retrieve the array containing the dictionaries with key value
    # pairs of the data we need to run the ID3 Also create a list of the file names to loop through them and run the
    # ID3 for each file # dataFiles = ["candy.data", "house_votes_84.data", "tennis.data"]
    dataFiles = ["house_votes_84.data"]
    for file in dataFiles:
        # load the dataset from the file into the list of dictionaries using parse
        # 2) We got the dataset in the form of a list [ dict , dict , dict ] where dict contains key pair values.
        listOfDictionaries = parse(file)
        # Clean the dataset
        # Use this clean up function to remove all the "?" from the data and clean it up
        # listOfDictionaries = data_cleaning.cleanUpData(listOfDictionaries)
        # Use this clean up function to replace all the "?" from the data with the most probable values
        listOfDictionaries = data_cleaning.mostProbableValue(listOfDictionaries)
        # print(listOfDictionaries)
        # We are going to use Max IG as the root node to start building ID3
        # call the ID3 algorithm with the dataframe and the root node which is the Max IG
        tree = ID3(listOfDictionaries, 'republican')
        # tree is now a node (The Root node)
        print("\n")
        print("**********TREE BEFORE PRUNING**********")
        printDepthNode(tree, 0)
        print("\n")
        print("**********TREE TESTING RESULTS BEFORE PRUNING**********")
        print("Accuracy: ", test(tree, listOfDictionaries))
        print("\n")
        print("**********TREE AFTER PRUNING**********")
        tree = prune(tree, listOfDictionaries)
        printDepthNode(tree, 0)
        print("\n")
        print("**********TREE TESTING RESULTS AFTER PRUNING**********")
        print("Accuracy: ", test(tree, listOfDictionaries))
        print("All children: ", global_variables.children, " and after massive killing: ",
              global_variables.children - global_variables.children_kill_count)

        # Here we actually call the functions to train, test and validate the tree
        ac = accuracy.accuracy(listOfDictionaries)
        av_ac = [accuracy.runningAverage(ac[0]), accuracy.runningAverage(ac[1])]
        graph.graphing(av_ac[0], av_ac[1])

        '''
        # Tests related to the final version
        unit_tests.testID3AndEvaluate()
        unit_tests.testPruning()
        unit_tests.testID3AndTest()
        unit_tests.testPruningOnHouseData(file)
        '''


if __name__ == "__main__":
    main()

'''
def prune_reduce_error(node):
    parentNode = node
    stack = [node]

    while stack:
        node = stack.pop()
        children_to_kill = []
        #
        # Will simply run through the tree and delete any nodes whose children are all leaf nodes
        for child in node.children:
            # changing entropy here
            if node.children[child].label != "leaf":
                if node.children[child].information_gain <= 0:
                    children_to_kill.append(child)
                else:
                    # if it is not a leaf, append it cause we have to visit it and get it's children
                    stack.append(node.children[child])

        if len(children_to_kill) > 0:
            for child_to_kill in children_to_kill:
                del node.children[child_to_kill]
            # assign label of new pruned node as leaf, cause it does not have children anymore
            node.label = "leaf"

    return parentNode
'''

'''

def prune(node, examples):

    """
        Takes in a trained tree and a validation set of examples. Prunes nodes in order
        to improve accuracy on the validation data; the precise pruning strategy is up to you.
        """
    parentNode = node
    stack = [node]

    while stack:
        node = stack.pop()
        children_count = len(node.children)
        highest_ig = 0
        highest_ig_value = None
        # Will simply run through the tree and delete any nodes whose children are all leaf nodes
        for child in node.children:
            # changing entropy here
            if node.children[child].label == "leaf":
                children_count -= 1
                global_variables.children_kill_count += 1
            else:
                # if it is not a leaf, append it cause we have to visit it and get it's children
                stack.append(node.children[child])
                if node.children[child].information_gain > highest_ig:
                    highest_ig = node.children[child].information_gain
                    highest_ig_value = node.children[child].value

        if children_count == 0:
            # empty the children
            node.children = {}
            # assign label of new pruned node as leaf, cause it does not have children anymore
            node.label = "leaf"
            node.value = highest_ig_value
            # Uncomment this line is we want aggressive pruning
            # will delete the whole parent node and not just the children
            # del node

    return parentNode

'''

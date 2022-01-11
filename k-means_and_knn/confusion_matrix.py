# target is the list of actual labels, predictions is the labels that we predicted. The diagonal of the confusion
# matrix displays the count, in which predictions and labels correspond.
def confusion_matrix(targets, predictions):
    confusion_matrix_temp = {}
    clas = set(targets)
    # initializing and creating the confusion matrix
    for i in range(len(clas)):
        confusion_matrix_temp[i] = {i: 0 for i in range(len(clas))}

    for i in range(len(targets)):
        confusion_matrix_temp[targets[i]][predictions[i]] += 1
    return confusion_matrix_temp


def pretty_print(matrix):
    print("The confusion matrix is:\n")
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], sep="\t", end="\t", flush=True)
        print("\n")

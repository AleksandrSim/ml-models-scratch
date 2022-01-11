def dimensionality_reduction(list_of_sets):
    # We will reduce the number of columns on each row by removing redundant 0 values
    # To do that, we need to find the row with the maximum number of columns
    # this will give us the number of columns for row 1 without accounting the label
    print("The dimensions for each of the sets are: ", len(list_of_sets[0][0][1]))
    indexes = zero_values_index_locator_train_test(list_of_sets[0], list_of_sets[1])
    print("We are removing: ", len(indexes), " dimensions from each one of the sets")
    for list_set in list_of_sets:
        for index in sorted(indexes, reverse=True):
            for row in range(len(list_set)):
                del list_set[row][1][index]
        #print("The dimensions for each set after the reduction are: ", len(list_set[0][1]))
    return list_of_sets

def list_dimensionality_reduction(indexes, list_to_reduce):
    # We will reduce the number of columns on each row by removing redundant 0 values
    # To do that, we need to find the row with the maximum number of columns
    # this will give us the number of columns for row 1 without accounting the label
    print("The dimensions for each of the sets are: ", len(list_to_reduce[0][1]))
    print("We are removing: ", len(indexes), " dimensions from each one of the sets")
    for index in sorted(indexes, reverse=True):
        for row in range(len(list_to_reduce)):
            del list_to_reduce[row][1][index]
    print(
        "The dimensions for each set after the reduction are: ",
        len(list_to_reduce[0][1]),
    )
    return list_to_reduce

def list_dimensionality_reduction(indexes, list_to_reduce):
    # We will reduce the number of columns on each row by removing redundant 0 values
    # To do that, we need to find the row with the maximum number of columns
    # this will give us the number of columns for row 1 without accounting the label
    print("The dimensions for each of the sets are: ", len(list_to_reduce[0][1]))
    print("We are removing: ", len(indexes), " dimensions from each one of the sets")
    for index in sorted(indexes, reverse=True):
        for row in range(len(list_to_reduce)):
            del list_to_reduce[row][1][index]
    print(
        "The dimensions for each set after the reduction are: ",
        len(list_to_reduce[0][1]),
    )
    return list_to_reduce


def dimensional_smash(indexes, set_of_lists):
    print("The dimensions for each of the sets are: ", len(set_of_lists[0][0][1]))
    # print("We are removing: ", len(indexes), " dimensions from each one of the sets")
    for i in range(0, len(set_of_lists)):
        set_of_lists[i] = byebyeboring(indexes, set_of_lists[i])

    print(
        "The dimensions for each set after the reduction are: ",
        len(set_of_lists[0][0][1]),
    )
    return set_of_lists


def zero_values_index_locator(data_set):
    # this will give us the number of columns for row 1 without accounting the label
    max_columns = len(data_set[0][1])
    column_index_zero_values = []

    for column in range(max_columns):
        # declare a value (boolean) which will check if all the rows for this column value are all 0
        all_zeros = True
        for row in range(len(data_set)):
            # Go into each column, after skip each line to see if this column's rows are all 0
            if int(data_set[row][1][column]) != 0:
                # check if the value of the specific column is 0
                all_zeros = False
                break
        if all_zeros:
            # if this column's values are all 0 append the index of this column
            column_index_zero_values.append(column)

    return column_index_zero_values


def zero_values_index_locator_train_test(train_set, test_set):
    # this will give us the number of columns for row 1 without accounting the label
    max_columns = len(train_set[0][1])
    column_index_zero_values = []

    for column in range(max_columns):
        # declare a value (boolean) which will check if all the rows for this column value are all 0
        all_zeros = True
        for row in range(len(test_set)):
            # Go into each column, after skip each line to see if this column's rows are all 0
            if int(train_set[row][1][column]) != 0:
                # check if the value of the specific column is 0
                all_zeros = False
                break
        if all_zeros:
            # if this column's values are all 0 append the index of this column
            column_index_zero_values.append(column)

    return column_index_zero_values


def byebyeboring(boring, dataset):
    output = []
    dim = len(boring)
    length = len(dataset)

    for i in range(length):
        new_point = [dataset[i][0], []]
        for j in range(dim):
            if boring[j] == 1:
                new_point[1].append(dataset[i][1][j])
        output.append(new_point)

    return output


def binary_conversion(data_sets):
    # We will reduce the number of columns on each row by removing redundant 0 values
    # To do that, we need to find the row with the maximum number of columns
    for data_set in data_sets:
        max_columns = len(data_set[0][1])
        for row in range(len(data_set)):
            for column in range(max_columns):
                if data_set[row][1][column] != 0:
                    # turn it to binary
                    data_set[row][1][column] = 1
    return data_sets


"""
# returns the columns with useful data
def the_zero_sum_game(dataset):
    running_total = []
    for i in range(len(dataset)):
        if i == 0:
            running_total = dataset[i][1]
        else:
            running_total = sum_lists(running_total, dataset[i][1])
    return running_total
def zero_sum_manifest(useful, dataset):
    output = []
    for i in range(len(dataset)):
        new_point = [dataset[i][0], []]
        for j in range(len(useful)):
            if useful[j] != 0:
                new_point[1].append(dataset[i][1][j])
        output.append(new_point)
    return output
def beware_deviants(useful, dataset):
    output = []
    mean = helper.mean_of_list(useful)
    std_dev = helper.std_dev(helper.var_of_list(useful, mean))
    for i in range(len(dataset)):
        new_point = [dataset[i][0], []]
        for j in range(len(useful)):
            if useful[j] / std_dev > 1.0:
                new_point[1].append(dataset[i][1][j])
        output.append(new_point)
    return output
"""

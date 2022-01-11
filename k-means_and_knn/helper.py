from math import sqrt


def sum_lists(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


def mean_of_list(a):
    length = len(a)
    sum = 0
    for i in a:
        sum += i

    return sum / length


def var_of_list(a, mean):
    length = len(a)
    sum = 0

    for i in a:
        sum += (i - mean) ** 2
    return sum / length


def std_dev(var):
    return sqrt(var)


def matrix_vec_mult(a, b):
    # i x j * n x m = i x m only works if j == n

    rows_a = len(a)
    rows_b = len(b)

    cols_a = len(a[0])

    result = []

    if cols_a != rows_b:
        print("Size mismatch")
        return

    for i in range(rows_a):

        sum = 0
        for j in range(cols_a):
            sum += a[i][j] * b[j]
        result.append(sum)

    return result


# def dot_product


def zero_vector(dim):
    output = []
    for i in range(dim):
        output.append(0)

    return output


def one_vector(dim):
    output = []
    for i in range(dim):
        output.append(1)
    return output

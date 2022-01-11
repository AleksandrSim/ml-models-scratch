import math


# returns Euclidean distance between vectors a and b
def euclidean(a, b):

    sums = 0.0
    for i in range(len(a)):
        sums += abs(a[i] - b[i]) ** 2

    return math.sqrt(sums)

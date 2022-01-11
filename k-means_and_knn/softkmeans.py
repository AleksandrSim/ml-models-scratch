import math

from euclidean_distance import euclidean
from cosine_similarity import cosim
from kmeans import random_centroids, remove_labels, apply_to_query
from math import exp
from helper import matrix_vec_mult, one_vector, zero_vector
import global_variables


def denom(hidden, centroid_label):
    sum = 0
    dim = len(hidden)

    for i in range(dim):
        sum += hidden[i][centroid_label]
    return sum


# ALEX
def generate_centroids(train, matrix, number_of_k):
    new_centroids = [[] for i in range(number_of_k)]

    for centroid_number in range(len(new_centroids)):

        for feature in range(len(train[0])):
            centroid_value = 0
            probabilities_sums = 0
            for row_number in range(len(train)):
                centroid_value += (
                    train[row_number][feature] * matrix[row_number][centroid_number]
                )
                probabilities_sums += matrix[row_number][centroid_number]
            final_value = centroid_value / probabilities_sums
            new_centroids[centroid_number].append(final_value)

    return new_centroids


# needs to be rewritten
# for a centroid add prob * point's value for every dimension
def additive_auxiliary(hidden, points, centroid_label):
    output = []
    for i in range(len(hidden)):
        for p in points:
            pass

    return output


def divise_auxiliary(cent, denominator):
    output = []
    for c in cent:
        output.append(c / denominator)


def soft_clusters_to_centers(centroids, points, hidden_matrix):
    new_centroids = []
    k = len(centroids)
    length = len(points)
    dim = len(points[0])

    ones = one_vector(k)

    for i in range(k):
        denominator = denom(hidden_matrix, i)
        new_cent = []

        # add_aux call HERE

        new_cent = divise_auxiliary(new_cent, denominator)
    pass


# returns the "force" of each relationship, based on
# e ^ (-stiffness * distance between point and centroid)
def gravitic_force(centroid, point, stiffness):

    return exp(-stiffness * math.sqrt(euclidean(centroid, point)))
    # w = exp(-stiffness * euclidean(centroid, point))
    # return exp(-stiffness * euclidean(centroid, point)) if w != 0 else 0.000000000000001


def hidden_matrix_element(centroids, point, stiffness, cent_index):
    sum = 0
    for centroid in centroids:

        sum += gravitic_force(centroid, point, stiffness)
    return gravitic_force(centroids[cent_index], point, stiffness) / sum


def hidden_matrix(centroids, points, stiffness):
    matrix = []
    for i in range(len(points)):
        matrix.append([])

        for j in range(len(centroids)):

            matrix[i].append(hidden_matrix_element(centroids, points[i], stiffness, j))
    return matrix


def soft_kmeans(train, test, metric):
    labels = []
    k = 10
    stiffness = 1
    max_iterations = 200
    iterations = 0
    data = remove_labels(train)[1]

    old_centroids = []
    new_centroids = random_centroids(train, k)

    # soft_clusters_to_centers(old_centroids, data, hidden)
    while iterations < max_iterations:
        old_centroids = new_centroids
        hidden = hidden_matrix(old_centroids, data, stiffness)

        new_centroids = generate_centroids(data, hidden, k)

        iterations += 1
        print(iterations)

    for i in range(k):
        labels.append(i)
    test_labels, test_points = remove_labels(test)
    test_new_labels = apply_to_query(
        new_centroids, labels, test_points, global_variables.metrics[0]
    )

    return test_new_labels, test_labels

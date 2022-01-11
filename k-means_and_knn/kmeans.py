import math
from euclidean_distance import euclidean
from cosine_similarity import cosim
from random import randrange
from helper import sum_lists


# given a list of centroids, label a point
# centroids is a list of centroids
# labels is the list of each centroid such that the label of centroids[i] is labels[i]
# point is the point to be labelled
# dist_type is the type of distance to be used in calcualtion
def label_point(centroids, c_labels, point, dist_type):
    # setup
    length = len(c_labels)
    min_dist = float("inf")
    min_label = None

    # check if distance between centroid and point is below minimum recorded distance
    # if so, change min_label and min_dist to show this
    for i in range(length):
        dist = float("inf")
        # print(centroids[i], point)
        if dist_type == "euclidean":
            dist = euclidean(centroids[i], point)
        elif dist_type == "cosine":
            dist = cosim(centroids[i], point)
        # print("dist from ", i, ":", dist, "cen:" , centroids[i],"point:" ,point)
        if dist < min_dist:
            min_label = c_labels[i]
            min_dist = dist
    # print(min_label)
    return min_label


# given a list of points determine new location for centroid
def centroid(points):
    if len(points) == 0:
        return None
    new_centroid = []
    # print(points)
    length = len(points)
    dim = len(
        points[0]
    )  # calculates the number of dimensions of every point in the points list
    for i in range(dim):
        new_centroid.append(0)
        # adds each value to centroid
        for p in points:
            new_centroid[i] += p[i]
        new_centroid[i] /= length  # finds the average
    return new_centroid


# compares old centroids to new centroids,
# if they are the same or max iterations has been reached return true, else false
def stop(old, new, i, n):
    return old == new or n > i


# Initialing random centroid based on number of features and k.
"""
def random_centroids(length_of_vector, k):
    random_centroids = []
    for i in range(k):
        cent = []
        for j in range(length_of_vector):
            rand = randrange(10)
            if rand <= 7:
                rand = 0
            else:
                rand = 1
            cent.append(rand)
        random_centroids.append(cent)
    return random_centroids
"""
# grabs a random point and creates a centroid upon it
def random_centroids(dataset, k):

    random_centroids = []

    length = len(dataset)
    window_size = int(length / k)
    i = 0

    while i < length:
        rand = randrange(0, window_size)
        # print(dataset[i + rand][1])
        random_centroids.append(dataset[i + rand][1])
        i += window_size
    return random_centroids


def apply_to_query(centroids, c_labels, query_wo_labels, metric):
    new_query_labels = []

    for point in query_wo_labels:
        new_query_labels.append(label_point(centroids, c_labels, point, metric))

    return new_query_labels


def remove_labels(dataset):
    labelless = []
    labels = []
    for v in dataset:
        labelless.append(v[1])
        labels.append(v[0])
    return labels, labelless


def add_label(vector, label):
    return [label, vector]


def kmeans(train, query, metric):
    k = 10
    max_iterations = 30
    iterations = 0

    num_points = len(train)
    dim = len(train[0][1])

    labels = []
    for i in range(k):
        labels.append(i)

    old_centroids = None
    org_labels, points = remove_labels(train)
    new_centroids = random_centroids(train, k)
    point_count = []

    while not stop(old_centroids, new_centroids, max_iterations, iterations):

        old_centroids = new_centroids
        new_centroids = []
        point_count = []
        # initializes variables for this step
        for i in range(k):
            cent = []
            for j in range(dim):
                cent.append(0)
            new_centroids.append(cent)
            point_count.append(0)

        # determines the label of a point and adds point to new centroids
        for p in points:
            label = label_point(old_centroids, labels, p, metric)
            point_count[label] += 1

            new_centroids[label] = sum_lists(new_centroids[label], p)

        # divides new centroids by the number of points associated with the centroid
        for i in range(k):

            if point_count[i] != 0:
                for j in range(dim):
                    new_centroids[i][j] /= point_count[i]
            else:
                new_centroids[i] = old_centroids[i]

        iterations += 1
        print(iterations)

        output = []
    for p in points:
        label = label_point(new_centroids, labels, p, metric)
        output.append(label)
        # output.append(add_label(p, label))

    query_labels, query_points = remove_labels(query)
    query_new_labels = apply_to_query(new_centroids, labels, query_points, metric)

    return query_new_labels, query_labels


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def old_kmeans(train, query, metric):
    # hyperparameters
    k = 10
    max_iterations = 100
    iterations = 0
    old_centroids = None

    labels = []
    for i in range(k):
        labels.append(i)

    # length of each vector in the dataset
    length_of_vector = len(train[0][1])
    # length of the total dataset (eg. how many numbers)
    length_of_dataset = len(train)
    # Initialing random centroid
    new_centroids = random_centroids(length_of_vector, k)
    points = remove_labels(train)
    point_labels = []
    while not stop(old_centroids, new_centroids, max_iterations, iterations):
        # reset the old_centroid
        # print(old_centroids," - > " ,new_centroids)
        if iterations != 0:
            for i in range(len(old_centroids)):
                pass
                # print(euclidean(old_centroids[i], new_centroids[i]))
        old_centroids = new_centroids

        # print(old_centroids)
        # vector which holds the different labels we collect
        point_labels = []
        # print(length_of_vector)

        # determine labels for each "number" in the dataset with min distance
        for p in points:
            point_labels.append(label_point(new_centroids, labels, p, metric))
        # print(old_centroids, point_labels, points)
        # print("")
        # Get the new centroids based on new label
        new_centroids = []
        for i in range(k):
            point_assoc = []
            for j in range(len(points)):
                if point_labels[j] == i:
                    point_assoc.append(points[j])
            # print(point_assoc)
            # print(i, "points", len(point_assoc))
            c = centroid(point_assoc)
            # print(c, point_assoc)
            if c is not None:
                new_centroids.append(c)
            else:
                new_centroids.append(old_centroids[i])

        # print(new_centroids)
        iterations += 1
        print(iterations)
    # unite the data
    output = []
    for i in range(len(points)):
        # print(point_labels[i])
        output.append(add_label(points[i], point_labels[i]))

    return output

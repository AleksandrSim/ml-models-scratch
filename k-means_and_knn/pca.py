# I'm going to hate myself for this...
import helper


def cov(dataset, mean_x, mean_y, x, y):
    length = len(dataset)
    sum = 0
    for i in range(length):
        # for every single point calulate (Xi - meanX) (Yi - meanY) and add to sum
        sum += (dataset[i][1][x] - mean_x) * (dataset[i][1][y] - mean_y)

    return sum / length


def var_of_dim(dataset, dim):
    values_of_dimension = []
    length = len(dataset)

    for i in range(length):
        values_of_dimension.append(dataset[i][1][dim])

    return helper.var_of_list(
        values_of_dimension, helper.mean_of_list(values_of_dimension)
    )


def covariance_matrix(dataset):
    length = len(dataset)
    dim = len(dataset[0][1])

    means = []
    print("calculating means...")
    # calculate the means of every column in the dataset
    for j in range(dim):
        # start all means at 0 and loop over dataset, adding all values of each column j
        means.append(0)
        for i in range(length):
            means[j] += dataset[i][1][j]  # what the hell is this structure?
        # divide the sums by the number of points or length
        means[j] /= length

    matrix = []
    print("creating covariance matrix...")
    # for each possible point calculate the covariance
    for i in range(dim):
        print("\t", i, "of", dim)
        matrix.append([])
        for j in range(dim):
            matrix[i].append(cov(dataset, means[i], means[j], i, j))

    return matrix


def pca(dataset):

    # dataset should be normalized...

    # create covariance matrix, in our case should be 784 x 784

    # determine the eigenvectors and eigenvalues of covariance matrix

    # take the top (based on eigenvalues) p eigenvectors
    # should be possible to determine a threshold on eigenvalues

    # create feature vector from remaining eigenvectors
    # resultant dataset = transpose of feature vector * transpose of original dataset

    pass

    # nope, won't work

import unittest
import numpy as np
import euclidean_distance
import cosine_similarity
import kmeans
import global_variables
from scipy import spatial
import helper
from helper import one_vector, zero_vector
import starter

# from sklearn.neighbors import kneighborsClassifier
# from sklearn.cluster import kmeans
import random

vec1 = [1, 2]
vec2 = [3, 4]
vec3 = [5, 6]

vec_c = [0.65, 0.8, 0.85, 0.7, 0.65]
vec_p = [1, 1, 1, 0, 1]


def random_points(dim, num):
    points = []

    for i in range(num):
        points.append([i, []])
        for j in range(dim):
            rand = random.randrange(1000)
            if rand < 230:
                points[i][1].append(0)
            else:
                points[i][1].append(1)
    return points


class Tests(unittest.TestCase):
    def test_is_it_working(self):
        assert True

    def test_euclidean_dist_1(self):
        dist2 = euclidean_distance.euclidean(vec1, vec2)
        array_1 = np.array(vec1)
        array_2 = np.array(vec2)
        dist = np.linalg.norm(array_1 - array_2)
        print(dist, dist2)
        assert dist == dist2

    def test_euclidean_dist_2(self):
        one = one_vector(784)
        zero = zero_vector(784)

        a_1 = np.array(one)
        a_0 = np.array(zero)

        dist1 = euclidean_distance.euclidean(one, zero)
        dist2 = np.linalg.norm(a_1 - a_0)

        print(dist1, dist2)
        assert dist1 == dist2

    def test_cosine_sim_1(self):
        sim1 = cosine_similarity.cosim(vec1, vec2)
        sim2 = 1 - spatial.distance.cosine(vec1, vec2)
        assert sim1 == sim2

    def test_cosine_sim_2(self):
        sim1 = cosine_similarity.cosim(vec2, vec3)
        sim2 = 1 - spatial.distance.cosine(vec2, vec3)
        assert sim1 == sim2

    def test_of_sanity(self):
        sim1 = cosine_similarity.cosim(vec_c, vec_p)
        sim2 = 1 - spatial.distance.cosine(vec_c, vec_p)
        print(sim1, sim2)
        assert sim1 == sim2

    def test_kmean(self):
        points = random_points(9, 20)
        # print(points)
        # print(kmeans.kmeans(points,  None, "cosine"))
        print(kmeans.kmeans(points, None, global_variables.metrics[0]))

    # def test_accuracy(self):
    #     input = starter.main().data_binary_train
    #     output = starter.main().clustering_result
    #     print(input)

    def test_matrixmul(self):

        m = []
        v = []
        m_r = 10
        m_c = 10
        for i in range(m_r):
            m.append([])
            for k in range(m_c):
                m[i].append(i + k)

        for i in range(m_r):
            v.append(i)

        print(helper.matrix_vec_mult(m, v))

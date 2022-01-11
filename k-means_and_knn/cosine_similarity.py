import math


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    numerator = sums_a = sums_b = 0.0
    for i in range(len(a)):
        numerator += a[i] * b[i]
        sums_a += a[i] ** 2
        sums_b += b[i] ** 2
    return numerator / (math.sqrt(sums_a) * math.sqrt(sums_b))

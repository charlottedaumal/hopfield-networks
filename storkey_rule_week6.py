import numpy as np
import random as rd


def storkey_weights(patterns):
    dimension = patterns.shape[1]
    nb_patterns = patterns.shape[0]
    w = np.zeros(dimension**2).reshape(dimension, dimension)
    for mu in range(0, nb_patterns):
        new_w = np.zeros(dimension**2).reshape(dimension, dimension)
        for i in range(0, dimension):
            for j in range(0, dimension):
                h = np.zeros(dimension**2).reshape(dimension, dimension)
                for k in range(0, dimension):
                    if (k != i) and (k != j):
                        product = w[i][k] * patterns[mu][k]
                        h[i][j] += product
                product_patterns = patterns[mu][i] * patterns[mu][j]
                sub_products_pattern_h = patterns[mu][i] * h[j][i] - patterns[mu][j] * h[i][j]
                new_w[i][j] = w[i][j] + 1/dimension * (product_patterns - sub_products_pattern_h)
        w = new_w
    return w

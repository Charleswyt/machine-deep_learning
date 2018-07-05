#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.07.05
Finished on  2018.07.05
@author: Wang Yuntao
"""

from math import *
import numpy as np

"""
    calculate all kinds of distance between two vectors
    euclidean_distance(x, y, digit=3)                       calculate euclidean distance of vector x and vector y
    manhattan_distance(x, y, digit=3)                       calculate manhattan distance of vector x and vector y
    chebyshev_distance(x, y, digit=3)                       calculate chebyshev distance of vector x and vector y
    minkowski_distance(x, y, p_value, digit=3)              calculate minkowski distance of vector x and vector y
    cosine_similarity(x, y, digit=3)                        calculate cosine  similarity of vector x and vector y
    jaccard_similarity(x, y, digit=3)                       calculate jaccard similarity of vector x and vector y
    standard_euclidean_distance(x, y, digit=3)              calculate standard euclidean distance of vactor x and vector y
    mahalanobi_distance(x, y, digit=3)                      calculate mahalanobi distance of vactor x and vector y
    pearson_correlation(x, y, digit=3)                      calculate pearson correlation of vactor x and vector y
    hamming_distance(x, y)                                  calculate hamming distance of vactor x and vector y
    bray_curtis_distance(x, y, digit=3)                     calculate bray curtis distance of vactor x and vector y
    nth_root(value, n_root, digit=3)                        calculate nth order root of value
    square_rooted(value, digit=3)                           calculate square root of value

Reference:
    https://www.cnblogs.com/denny402/p/7027954.html
    http://www.cnblogs.com/denny402/p/7028832.html
"""


def euclidean_distance(x, y, digit=3):
    """
    calculate euclidean distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        euclidean distance of two vector
    """
    return round(sqrt(sum(pow(xi - yi, 2) for xi, yi in zip(x, y))), digit)


def manhattan_distance(x, y, digit=3):
    """
    calculate manhattan distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        manhattan distance of two vector
    """
    return round(sum(abs(xi - yi) for xi, yi in zip(x, y)), digit)


def nth_root(value, n_root, digit=3):
    """
    calculate nth root of value
    :param value: the value to be processed
    :param n_root: the order of root
    :param digit: retained digits after a decimal point, default is 3
    """
    root_value = 1 / float(n_root)
    return round(value ** root_value, digit)


def chebyshev_distance(x, y, digit=3):
    """
    calculate chebyshev distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        chebyshev distance of two vector
    """
    return round(max(abs(xi - yi) for xi, yi in zip(x, y)), digit)


def minkowski_distance(x, y, p_value, digit=3):
    """
    calculate minkowski distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        manhattan distance of two vector
    """
    return nth_root(sum(pow(abs(xi - yi), p_value) for xi, yi in zip(x, y)), p_value, digit)


def square_rooted(value, digit=3):
    """
    calculate the square root of vector x
    :param value: the value to be processed
    :param digit: retained digits after a decimal point, default is 3
    :return:
        the square root of vector x
    """
    return round(sqrt(sum([xi * xi for xi in x])), digit)


def cosine_similarity(x, y, digit=3):
    """
    calculate cosine similarity of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        cosine similarity of two vector
    """
    numerator = sum(xi * yi for xi, yi in zip(x, y))
    denominator = square_rooted(x, digit) * square_rooted(y, digit)

    return round(numerator / denominator, digit)


def jaccard_similarity(x, y, digit=3):
    """
    calculate jaccard similarity of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        jaccard similarity of two vector
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    
    return round(intersection_cardinality / union_cardinality, digit)


def standard_euclidean_distance(x, y, digit=3):
    """
    calculate standard euclidean distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        standard euclidean distance of two vector
    """
    x_mean, x_var = np.mean(x), np.var(x)
    y_mean, y_var = np.mean(y), np.var(y)
    x_std, y_std = (x - x_mean) / x_var, (y - y_mean) / y_var
    return round(sqrt(sum(pow(xi - yi, 2) for xi, yi in zip(x_std, y_std))), digit)


def mahalanobi_distance(x, y, digit=3):
    """
    calculate mahalanobi distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        mahalanobi distance of two vector
    """
    merge = np.vstack([x, y])
    merge_transposition = merge.T
    covariance_matrix = np.cov(merge)
    distances = []
    try:
        covariance_matrix_inverse = np.linalg.inv(covariance_matrix)
        dimension = merge_transposition.shape[0]
        for i in range(0, dimension):
            for j in range(i + 1, dimension):
                delta = merge_transposition[i] - merge_transposition[j]
                distance = np.sqrt(
                    np.dot(np.dot(delta, covariance_matrix_inverse), delta.T))
                distances.append(round(distance, digit))
    except np.linalg.linalg.LinAlgError:
        print("Singular matrix error.")

    return distances


def pearson_correlation(x, y, digit=3):
    """
    calculate pearson correlation of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :param digit: retained digits after a decimal point, default is 3
    :return:
        pearson correlation of two vector
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sqrt(sum(pow(xi - x_mean, 2) for xi in x)) * sqrt(sum(pow(yi - y_mean, 2) for yi in y))

    return round(numerator / denominator, digit)


def hamming_distance(x, y):
    """
    calculate haming distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :return:
        haming distance of two vector
    """
    sub_list = list((xi - yi) for xi, yi in zip(x, y))
    
    return len(x) - sub_list.count(0)


def bray_curtis_distance(x, y, digit=3):
    """
    calculate Bray Curtis distance of vactor x and vector y
    :param x: vector x
    :param y: vector y
    :return:
        Bray Curtis distance of two vector
    """
    numerator = sum(abs(xi - yi) for xi, yi in zip(x, y))
    denominator = sum(x) + sum(y)

    return round(numerator / denominator, digit)


if "__main__" == __name__:
    x = [3, 4, 5]
    y = [4, 5, 6]

    dis_euclidean = euclidean_distance(x, y)
    dis_manhattan = manhattan_distance(x, y)
    dis_chebyshev = chebyshev_distance(x, y)
    dis_minkowski = minkowski_distance(x, y, 3)
    cosine_sim = cosine_similarity(x, y)
    jaccard_sim = jaccard_similarity(x, y)
    dis_std_euclidean = standard_euclidean_distance(x, y)
    dis_mahalanobi = mahalanobi_distance(x, y)
    pearson_corr = pearson_correlation(x, y)
    dis_hamming = hamming_distance(x, y)
    dis_bray_curtis = bray_curtis_distance(x, y)

    print("vector x: ", x)
    print("vector y: ", y)
    print("========================================")
    print("euclidean distance: %.3f" % dis_euclidean)
    print("manhattan distance: %.3f" % dis_manhattan)
    print("chebyshev distance: %.3f" % dis_chebyshev)
    print("minkowski distance: %.3f, order: 3" % dis_minkowski)
    print("cosine similarity: %.3f" % cosine_sim)
    print("jaccard similarity: %.3f" % jaccard_sim)
    print("stadard euclidean distance: %.3f" % dis_std_euclidean)
    print("mahalanobi distance: ", dis_mahalanobi)
    print("pearson correlation: %.3f" % pearson_corr)
    print("hamming distance: %d" % dis_hamming)
    print("bray curtis distance: %.3f" % dis_bray_curtis)

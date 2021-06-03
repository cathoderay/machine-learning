"""
    Multivariate Linear Regression from scratch.

    This module is a self imposed exercise to 
    understand how linear regression works.

    In this particular implementation I chose 
    the Gradient Descent Method.

    Since it's easier to understand its results
    with 2 variables (R^2), you can see an
    application of it with a simple dataset in
    application.py 
"""


__author__ = "Ronald Kaiser"
__email__  = "raios dot catodicos at gmail dot com"


from functools import partial
from random import random


def linear_function(p, x):
    return sum(pi * xi for pi, xi in zip(p, x))


def mean_square(f, data):
    return sum((f(x) - y) ** 2 for x, y in data) / len(data)


def cost(f, data):
    return mean_square(f, data) / 2


def partial_derivative(lf, i, data):
    return sum((lf(x) - y) * x[i] for x, y in data) / len(data)


def gradient_descent(data, threshold=0.000001, max_steps=1000, learning_rate=1):
    n = len(data[0][0])
    p = [random()] * n
    prediction = partial(linear_function, p)
    error = cost(prediction, data)
    last_error = error + threshold + 1
    steps = 0

    while abs(last_error - error) > threshold and steps < max_steps:
        p = [p[i] - (learning_rate) * partial_derivative(prediction, i, data) for i in range(n)]
        steps += 1
        prediction = partial(linear_function, p)
        last_error, error = error, cost(prediction, data)
        learning_rate = learning_rate if error < last_error else learning_rate/3
        steps = steps if error < last_error else 0
        if learning_rate == 0: break
        print(f"Error: {error}, Alpha: {learning_rate}, Steps: {steps}, Parameters: {p}")
    return p


def test_linear_func():
    assert linear_function([2, 1], [1, 3]) == 2 * 1 + 1 * 3
    assert linear_function([5, 6], [1, 3]) == 5 * 1 + 6 * 3


def test_get_cost_perfect_fit():
     data = [([1, 1], 3), ([1, 3], 4), ([1, 5], 5)]
     params = [2.5, 0.5]
     hypothesis = partial(linear_function, params)
     assert cost(hypothesis, data) == 0


def test_get_cost_wrong_fit():
     data = [([1, 1], 3), ([1, 3], 4), ([1, 5], 5)]
     params = [3, 0.5]
     hypothesis = partial(linear_function, params)
     assert cost(hypothesis, data) != 0


def test_get_cost_comparing_two_hypothesis():
    data = [([1, 1], 3), ([1, 3], 4), ([1, 5], 5)]

    params = [3, 0.5]
    hypothesis = partial(linear_function, params)
    v1 = cost(hypothesis, data)

    params = [50, 300]
    hypothesis = partial(linear_function, params)
    v2 = cost(hypothesis, data)

    assert v1 < v2


def test_gradient_descent():
     data = [([1, 1], 3), ([1, 3], 4), ([1, 5], 5)]
     learned_params = gradient_descent(data)
     learned_hypothesis = partial(linear_function, learned_params) 

     assert cost(learned_hypothesis, data) < 0.0001

#TODO: add more tests

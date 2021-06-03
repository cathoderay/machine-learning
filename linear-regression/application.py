import csv
from functools import partial

import matplotlib.pyplot as plt

from linear_regression import gradient_descent, linear_function, cost


def read_csv(file):
    data = []
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0].startswith('X'): continue
            datum = ([1, float(row[0])], float(row[1]))
            data.append(datum)
    return data


# plot original dataset
data = read_csv('dataset.csv')
x = [x[1] for x, y in data]
y = [y for x, y in data]
plt.plot(x, y, marker='x', linestyle='None')


# plot learned hypothesis
parameters = gradient_descent(data, max_steps=100000000)
hypothesis = partial(linear_function, parameters)
y = [hypothesis(x) for x, y in data]
plt.plot(x, y, marker='x')
print(f"Solution: {parameters}")
print(f"Hypothesis cost: {cost(hypothesis, data)}")
plt.show()

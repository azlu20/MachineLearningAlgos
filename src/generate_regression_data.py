import random

import numpy as np

def generate_regression_data(degree, N, amount_of_noise=1.0):
    x = []
    for i in range(N):
        x.append(random.random()*2-1)
    x = np.array(x)
    degrees = []
    for i in range(degree):
        degrees.append(random.random()*20-10)
    f = np.poly1d(degrees)
    y = []
    for j in range(len(x)):
        y.append(f(x[j]))
    gauss_noise = np.random.normal(0, amount_of_noise*np.std(y), len(y))
    y = np.add(gauss_noise, y)
    return x, y
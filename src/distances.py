import math

import numpy as np

def euclidean_distances(X, Y):
    D = []
    for i in X:
        temp = []
        for j in Y:
            k = 0
            total = 0
            while(k<len(i)):
                total += (i[k] - j[k])**2
                k+=1
            temp.append(math.sqrt(total))
        D.append(temp)
    return np.asmatrix(D)




def manhattan_distances(X, Y):
    D = []
    for i in X:
        temp = []
        for j in Y:
            k = 0
            total = 0
            while(k<len(i)):
                total += abs(i[k] - j[k])
                k+=1
            temp.append(total)
        D.append(temp)
    return np.asmatrix(D)


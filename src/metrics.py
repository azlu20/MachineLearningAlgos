import numpy as np

def mean_squared_error(estimates, targets):


    tempval = np.subtract(estimates, targets)
    tempval = np.square(tempval)
    summed = np.sum(tempval)
    return summed/(len(estimates))
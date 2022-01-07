import math

import numpy as np
import pytest
from matplotlib import pyplot
from .distances import euclidean_distances, manhattan_distances

def mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator



    def fit(self, features, targets):
        self.features = features
        self.targets = targets
        

    def predict(self, features, ignore_first=False):

        anslist = np.empty(shape=[features.shape[0], self.targets.shape[1]])
        for i in range(features.shape[0]):
            test = []
            test.append(features[i])
            if(self.distance_measure == "euclidean"):
                dist = euclidean_distances(np.array(test), self.features)
            else:
                dist = manhattan_distances(np.array(test), self.features)
            dist = dist.tolist()
            dist = dist[0]

            if (self.n_neighbors >= len(dist)):
                closest_neighbors = range(len(dist))
            else:
                closest_neighbors = np.argpartition(dist, self.n_neighbors)
            arr = []
            total = len(closest_neighbors)
            if(ignore_first):
                for j in range(self.targets.shape[1]):
                    for k in range(1, self.n_neighbors+1):
                        arr.append(self.targets[closest_neighbors[k]][j])
                    if (self.aggregator == "mode"):
                        currlist.append(mode(np.array(arr))[0])
                    else:
                        if (self.aggregator == "mean"):
                            currlist.append(np.mean(np.array(arr)))
                        else:
                            currlist.append(np.median(np.array(arr)))

            else:
                currlist = []
                for j in range(self.targets.shape[1]):
                    arr = []
                    for k in range(self.n_neighbors):
                            arr.append(self.targets[closest_neighbors[k]][j])
                    if (self.aggregator == "mode"):
                        currlist.append(mode(np.array(arr))[0])
                    else:
                        if (self.aggregator == "mean"):
                            currlist.append(np.mean(np.array(arr)))
                        else:
                            currlist.append(np.median(np.array(arr)))

            anslist[i] = currlist

        return anslist




import json
import math

import numpy as np
import os
from matplotlib.colors import ListedColormap

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


def mode(a, axis=0):

    scores = np.unique(np.ravel(a))  # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis), axis)
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
            if (self.distance_measure == "euclidean"):
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
            if (ignore_first):
                for j in range(self.targets.shape[1]):
                    for k in range(1, self.n_neighbors + 1):
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


def load_json_data(json_path):


    with open(json_path, 'rb') as f:
        data = json.load(f)
    features = np.array(data[0]).astype(float)
    targets = 2 * (np.array(data[1]).astype(float) - 1) - 1

    return features, targets


if __name__ == "__main__":

    try:
        import matplotlib.pyplot as plt
    except:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    data_files = [
        os.path.join('..', 'data', x)
        for x in os.listdir(os.path.join('..', 'data'))
        if os.path.splitext(x)[1] == '.json'
    ]

    aggregators = ['mean', 'median', 'mode']
    distances = ['euclidean', 'manhattan']
    neighbors = [1, 3, 5 ,7]

    for data_path in data_files:
        # Load data and make sure its shape is correct
        features, targets = load_json_data(data_path)
        targets = targets[:, None]  # expand dims
        for n in neighbors:
            for d in distances:
                for a in aggregators:
                    # make model and fit
                    knn2 = KNearestNeighbor(n, distance_measure=d, aggregator=a)
                    knn2.fit(features, targets)
                    test = knn2.predict(features)
                    h = 10
                    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                         np.arange(x_min, x_max, h))
                    Z = []
                    for i in range(xx.shape[1]):
                        x = xx[:,i]
                        y = yy[:,i]
                        y = y.T
                        newfeats = np.stack([x,y])
                        newfeats = newfeats.T
                        knn = KNearestNeighbor(n, distance_measure=d, aggregator=a)
                        knn.fit(features, targets)
                        labels = knn.predict(newfeats)
                        Z.append(labels)
                    Z = np.array(Z)
                    Z = np.squeeze(Z, axis=2)
                    Z= Z.reshape(xx.shape)
                    plt.figure(figsize=(6, 4))
                    plt.contourf(yy, xx, Z, cmap = ListedColormap(['#b19cd9', '#FFA500', '#FFFFE0']))
                    plt.scatter(features[:, 0], features[:, 1], c=targets)
                    plt.title(data_path)
                    name = os.path.join('..', 'data'.format(data_path))
                    if(data_path == '..\\data\\clean-spiral.json'):
                        name += 'clean-spiral'
                    else:
                        name += 'noisy-linear'
                    name += str(n) + d + a + '.png'
                    plt.savefig(name)
                    print(features.shape, targets.shape, data_path)

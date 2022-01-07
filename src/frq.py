import math
import os
import random

import numpy as np


try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    D = []
    # if(len(X) == 1 and len(Y) == 1):
    #     temp = []
    #     for i in X:
    #         for j in Y:
    #             temp.append((i-j)**2)
    #         D.append(temp)

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
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
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
    """
    Copied from scipy.stats.mode.
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L361-L451

    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned.
    The bin-count for the modal bins is also returned.
    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    """
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
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3
        closest neighbors are:
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]
            ]
        And the aggregator is 'mean', applied along each dimension, this will return for
        that point:
            [
                [2, 3, 4]
            ]

        Hint: numpy has functions for computing the mean and median, but you can use the `mode`
              function for finding the mode.

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels.
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional!

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples,
                n_dimensions).
        """

        self.features = features
        self.targets = targets

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor.
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
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
def generate_regression_data(degree, N, amount_of_noise=1.0):
    """
    Generates data to test one-dimensional regression models. This function:

    1)  Generates explanatory variable x: a shape (N,) array that contains
        Floats chosen at random between -1 and 1.

    2)  Creates a polynomial function f() of degree 'degree'. The polynomial's
        Float coefficients are chosen uniformally at random between -10 and 10.

    3)  Generates response variable y: a shape (N,) array that contains f(x),
        where the ith element of y is calculated by applying f() to the ith
        element of x

    4)  Adds Gaussian noise n to y. Here mean(n) = 0 and standard deviation
        (notated std(n)) is: std(n) = 'amount_of_noise' * std(y) and mean 0
        (Hint...use np.random.normal to generate this noise)

    Args:
        degree (int): degree of polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N, which responds to x as a
                        polynomial of degree 'degree'.

    """
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

def test_polynomial_regression():
    degrees = range(8, 10)
    amounts = [10, 100, 1000, 10000]

    for degree in degrees:
        p = PolynomialRegression(degree)
        for amount in amounts:
            x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
            p.fit(x, y)
            y_hat = p.predict(x)
            mse = mean_squared_error(y, y_hat)
            assert (mse < 1e-1)
    return


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.

        This class takes as input "degree", which is the degree of the polynomial
        used to fit the data. For example, degree = 2 would fit a polynomial of the
        form:

            ax^2 + bx + c

        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf

        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval.
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np

            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.polynomial = None

    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.


        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """

        z = np.ones((features.shape[0], self.degree + 1))

        for i in range(features.shape[0]):
            for j in range(1, self.degree + 1):
                z[i][j] = np.power(features[i], j)
        z_transpose = z.T
        left = np.matmul(z_transpose, z)
        left = np.linalg.inv(left)
        right = np.matmul(left, z_transpose)
        self.w = np.matmul(right, targets)
        # w = w[:,0]
        # w = np.flip(w)
        self.polynomial = np.poly1d(self.w)

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        z = np.empty(shape=(features.shape[0], self.degree + 1))
        for i in range(features.shape[0]):
            for j in range(self.degree + 1):
                z[i][j] = features[i] ** j

        return np.matmul(z, self.w)

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        # predictions = self.predict(features)
        # predictions_sorted = np.argsort(predictions)
        features_sorted = []
        predictions_sorted = []
        features_sorted2 = np.argsort(features)
        predictions = self.predict(np.array(features))
        for ele in features_sorted2:
            features_sorted.append(features[ele])
            predictions_sorted.append(predictions[ele])
        # for ele in predictions_sorted:
        #     features_sorted.append(features[ele])
        # plt.figure()
        # plt.scatter(features, targets)
        # plt.plot(features_sorted, predictions_sorted)
        # plt.plot(features_sorted, predictions_sorted)
        # plt.savefig("frq2.jpg")
        return features_sorted, predictions_sorted

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        estimates(np.ndarray): the estimated values (should be the same shape as targets)
        targets(np.ndarray): the ground truth values

    Returns:
        MSE(int): mean squared error calculated by above equation
    """

    tempval = np.subtract(estimates, targets)
    tempval = np.square(tempval)
    summed = np.sum(tempval)
    return summed/(len(targets))


x, y = generate_regression_data(4, 100, 0.1)
split_a_train_x = x[:10]
split_a_test_x = x[10:]
split_b_train_x = x[:50]
split_b_test_x = x[50:]
split_a_train_y = y[:10]
split_a_test_y = y[10:]
split_b_train_y = y[:50]
split_b_test_y = y[50:]
degrees= range(10)

trainerrorarr = []
testerrorarr = []
mintraining = None
train_x = None
train_y = None
train_degree = None
mintesting = None
test_x = None
test_y = None
test_degree = None
degree_errors = []
# plt.figure()
# plt.scatter(split_b_train_x, split_b_train_y)

for degree in degrees:
    p = PolynomialRegression(degree)
    p.fit(split_b_train_x, split_b_train_y)
    b_y = p.predict(split_b_train_x)
    trainerror = mean_squared_error(split_b_train_y, b_y)
    if(mintraining == None or trainerror <= mintraining):
        mintraining = trainerror
        train_x, train_y = p.visualize(split_b_train_x, split_b_train_y)
        train_degree = degree
    trainerrorarr.append(trainerror)
    b_y_test = p.predict(split_b_test_x)
    testerror = mean_squared_error(split_b_test_y, b_y_test)
    degree_errors.append(testerror)
    if(mintesting == None or testerror <= mintesting):
        mintesting = testerror
        test_x, test_y = p.visualize(split_b_test_x, split_b_test_y)
        test_degree = degree
    testerrorarr.append(testerror)
# plt.savefig("testsklearn.jpg")
plt.figure()
plt.scatter(split_b_train_x, split_b_train_y)
plt.plot(train_x, train_y)
plt.plot(test_x, test_y)
plt.legend(["Train Error: " + str(train_degree), "Test Error: " + str(test_degree) ])
plt.savefig("frq2real.jpg")
plt.figure()
takelog = lambda x: math.log(x)
trainerrorarr = list(map(takelog, trainerrorarr))
testerrorarr = list(map(takelog, testerrorarr))
plt.plot(degrees, trainerrorarr)
plt.plot(degrees, testerrorarr)
plt.legend(["Train Error", "Test Error"])
plt.xlabel("Degrees")
plt.ylabel("Error")
plt.savefig("frq.jpg")

#Split A
for degree in degrees:
    p = PolynomialRegression(degree)
    p.fit(split_a_train_x, split_a_train_y)
    a_y = p.predict(split_a_train_x)
    trainerror = mean_squared_error(split_a_train_y, a_y)
    if(mintraining == None or trainerror <= mintraining):
        mintraining = trainerror
        train_x, train_y = p.visualize(split_a_train_x, split_a_train_y)
        train_degree = degree
    trainerrorarr.append(trainerror)
    a_y_test = p.predict(split_a_test_x)
    testerror = mean_squared_error(split_a_test_y, a_y_test)

    if(mintesting == None or testerror <= mintesting):
        mintesting = testerror
        test_x, test_y = p.visualize(split_a_test_x, split_a_test_y)
        test_degree = degree
    testerrorarr.append(testerror)
# plt.savefig("testsklearn.jpg")
print(degree_errors)
plt.figure()
plt.scatter(split_a_train_x, split_a_train_y)
plt.plot(train_x, train_y)
plt.plot(test_x, test_y)
plt.legend(["Train Error: " + str(train_degree), "Test Error: " + str(test_degree) ])
plt.savefig("frq2real_A.jpg")
#KNN Section
k_values = [1, 3, 5, 7, 9]
train_acc_list = []
test_acc_list = []
lowest_test_acc = None
lowest_split = None
lowest_k_value = None
train_b = []
train_b.append(split_a_train_x)
train_b = np.array(train_b)
train_b = train_b.T
train_b_y = []
train_b_y.append(split_a_train_y)
train_b_y = np.array(train_b_y)
train_b_y = train_b_y.T
test_a = []
test_a.append(split_a_test_x)
test_a = np.array(test_a)
test_a = test_a.T
test_a_y = []
test_a_y.append(split_a_test_y)
test_a_y = np.array((test_a_y))
test_a_y = test_a_y.T
for ele in k_values:
    knn = KNearestNeighbor(ele, "euclidean", "mode")
    knn.fit(train_b, train_b_y)
    train_values = knn.predict(train_b)
    test_values = knn.predict(test_a)
    train_acc = mean_squared_error(train_b_y, train_values)
    test_acc = mean_squared_error(test_a_y, test_values)
    train_acc_list.append((train_acc))
    test_acc_list.append(test_acc)
    if(lowest_test_acc == None or test_acc <= lowest_test_acc):
        lowest_test_acc = test_acc
        lowest_split = test_values
        lowest_k_value = ele
plt.figure()
plt.plot(k_values, train_acc_list)
plt.plot(k_values, test_acc_list)
plt.legend(["Train Error", "Test Error"])
plt.xlabel("K values")
plt.ylabel("Error")
plt.savefig("frq3.jpg")
plt.figure()
features_sorted = []
predictions_sorted = []
features_sorted2 = np.argsort(split_a_test_x)
for ele in features_sorted2:
    features_sorted.append(split_a_test_x[ele])
    predictions_sorted.append(lowest_split[ele])
plt.plot(features_sorted, predictions_sorted, 'r-')
plt.scatter(split_a_train_x, split_a_train_y)
plt.legend(["K Value: " + str(lowest_k_value)])
plt.savefig("frq4.jpg")

datasets = [
    os.path.join('data', x)
    for x in os.listdir('data')
    if os.path.splitext(x)[-1] == '.json'
]
#Make contour
aggregators = ['mean', 'mode', 'median']
distances = ['euclidean', 'manhattan']
for data_path in datasets:
    # Load data and make sure its shape is correct
    features, targets = load_json_data(data_path)
    targets = targets[:, None]  # expand dims
    for d in distances:
        for a in aggregators:
            # make model and fit
            knn = KNearestNeighbor(1, distance_measure=d, aggregator=a)
            knn.fit(features, targets)

            # predict and calculate accuracy
            labels = knn.predict(features)
            plt.figure()
            plt.scatter(features, targets)
            plt.savefig("test.jpg")
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        self.degree = degree
        self.polynomial = None
    
    def fit(self, features, targets):

        z = np.ones((features.shape[0], self.degree+1))

        for i in range(features.shape[0]):
            for j in range(1, self.degree+1):
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

        z = np.empty(shape=(features.shape[0], self.degree+1))
        for i in range(features.shape[0]):
            for j in range(self.degree+1):
                z[i][j] = features[i] ** j

        return np.matmul(z, self.w)



    def visualize(self, features, targets):

        self.fit(features, targets)
        predictions = self.predict(features)
        plt.figure()
        plt.plot(features, predictions)
        plt.scatter(features, targets)
        plt.savefig("test.jpg")

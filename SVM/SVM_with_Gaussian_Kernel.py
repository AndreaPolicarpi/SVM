

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers
            

def gaussian(x1, x2, s=5.0):
    return np.exp(-linalg.norm(x1-x2)**2 / (2 * (s ** 2)))

class SVM(object):

    def __init__(self):
        self.kernel = gaussian

    def train(self, X, y):
        
        n_samples, n_features = X.shape
        
        ###THIS IS THE MATRIX OF KERNEL PRODUCTS
        
        K = np.zeros((n_samples, n_samples))
        self._k = K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)


        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))


        # solve QP problem (NORM) to find optimized alphas
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        print(a < 0)

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        pos = np.arange(len(a))[sv]
        self.a = a[sv]
        print(self.a)
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))
        
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[pos[n],sv])
           
        self.b /= len(self.a)


    def project(self, X):
        
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))




def separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

def non_sep_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

def circle_data():
        mean1 = [-4, 0]
        mean2 = [4, 0]
        mean3 = [0, 4]
        mean4 = [0, -4]
        mean5 = [0, 0]
        cov = [[0.5,0.8], [0.8, 0.5]]
        X1 = np.random.multivariate_normal(mean1, cov, 10)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean2, cov, 10)))
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 10)))
        X1 = np.vstack((X1, np.random.multivariate_normal(mean4, cov, 10)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean5, cov, 40)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2



def plot_separation(X1_train, X2_train, clf):
        plt.scatter(X1_train[:,0], X1_train[:,1], c='g')
        plt.scatter(X2_train[:,0], X2_train[:,1], c='b')
        plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c='m',marker="*")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')



        
        
        
def test():
   Xtest = [[0,0]]
   Xtest[0][0] = float(input("enter X coordinate of a point to classify: "))
   Xtest[0][1] = float(input("enter Y coordinate of a point to classify: "))
   Ytest = svm.predict(Xtest)
   print(Ytest)
   if Ytest == 1:
       print('classification: 1 ')
   else:
       print('classification: -1 ')
   plt.scatter(Xtest[0][0], Xtest[0][1], c='r')



X1, y1, X2, y2 = circle_data()
plt.scatter(X1[:,0], X1[:,1], c='g')
plt.scatter(X2[:,0], X2[:,1], c='b')
plt.grid()
plt.show()

X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

svm = SVM()
svm.train(X, y)
plot_separation(X[y==1], X[y==-1], svm)

"""
# TEST 3 POINTS AND THEN PLOT THE SEPARATION CURVE
for i in range(3):
    test()
plt.show()
"""


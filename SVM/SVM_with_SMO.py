import numpy as np
import matplotlib.pyplot as plt

DATA_DIM = 50
MAX_NUM_ITER = 50
EPS = 0.001
C = 1
MIN_ALPHA_OPT = 0.0000001


def linear(x1, x2):
    return np.dot(x1, x2)


# function to generate data
def create_data():

    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1, 0.35], [0.5, 0.1]])
    X1 = np.random.multivariate_normal(mean1, cov, DATA_DIM)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, DATA_DIM)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


#######################################################
#  SVM WITH LINEAR KERNEL
#######################################################
class SVM:
    
    
    
    def __init__(self, x, y, kernel = linear):
        self.x = x
        self.y = y
        self.kernel = kernel
        self.alpha = np.mat (np.zeros ((np.shape(x)[0], 1 )))
        self.b = np.mat([[0]])
        i = 0
        while (i < MAX_NUM_ITER):
            if (self.perform_smo() == 0):
                i += 1
            else:
                i = 0
        self.w = self.calc_w(self.alpha, self.x, self.y)
        self.positions = np.where(self.alpha>0)[0]
        self.support_vectors = self.x[self.get_positions()]
        self.support_vectors_y = self.y[self.get_positions()]
        
        
    def perform_smo (self):
        num_alpha_pairs_opt = 0
        for i in range(np.shape(self.x)[0]):
            Ei = np.multiply(self.y, self.alpha).T * self.kernel(self.x , self.x[i].T) + self.b - self.y[i]
            if (self.check_if_alpha_violates_kkt(self.alpha[i], Ei)):
                j = self.select_alpha_J(i, np.shape(self.x)[0])
                Ej = np.multiply(self.y, self.alpha).T * self.kernel(self.x , self.x[j].T) + self.b - self.y[j]
                alphaIold = self.alpha[i].copy()
                alphaJold = self.alpha[j].copy()
                bounds = self.bound_alpha(self.alpha[i], self.alpha[j], self.y[i], self.y[j])
                ETA = 2.0 * self.kernel(self.x[i] , self.x[j].T) \
                    - self.kernel(self.x[i] , self.x[i].T) \
                    - self.kernel(self.x[j] , self.x[j].T)
                if bounds[0] != bounds[1] and ETA < 0:
                    if self.optimize_alpha_pair(i, j, Ei, Ej, ETA, bounds, alphaIold, alphaJold):
                        num_alpha_pairs_opt += 1
        return num_alpha_pairs_opt
    
    
    
    def optimize_alpha_pair(self, i, j, Ei, Ej, ETA, bounds, alphaIold, alphaJold):
        optimized = False
        self.alpha[j] -= self.y[j] * (Ei - Ej)  /  ETA
        self.saturate_alpha_j(j, bounds)
        if (abs(self.alpha[j] - alphaJold) >= MIN_ALPHA_OPT):
            self.optimize_alphai_same_as_alphaj_opposite_direction(i, j, alphaJold)
            self.optimize_b(Ei, Ej, alphaIold, alphaJold, i, j)
            optimized = True
        return optimized
    
    
    
    def optimize_b(self, Ei, Ej, alphaIold, alphaJold, i, j):
        b1 = self.b - Ei - self.y[i]*(self.alpha[i]-alphaIold)*self.kernel(self.x[i],self.x[i].T) \
                          - self.y[j]*(self.alpha[j]-alphaJold)*self.kernel(self.x[i],self.x[j].T) 
        b2 = self.b - Ej - self.y[i]*(self.alpha[i]-alphaIold)*self.kernel(self.x[i],self.x[j].T) \
                          - self.y[j]*(self.alpha[j]-alphaJold)*self.kernel(self.x[j],self.x[j].T)  
        if (0 < self.alpha[i]) and (C > self.alpha[i]): self.b = b1
        elif (0 < self.alpha[j]) and (C > self.alpha[j]): self.b = b2
        else: self.b = (b1 + b2 ) / 2.0    
    
    
    
    def select_alpha_J(self, indexOf1stAlpha, numbOfRows):
        indexOf2ndAlpha = indexOf1stAlpha
        while (indexOf1stAlpha == indexOf2ndAlpha): 
            indexOf2ndAlpha = int (np.random.uniform(0, numbOfRows))
        return indexOf2ndAlpha
    
    
    
    def optimize_alphai_same_as_alphaj_opposite_direction(self, i, j, alphaJold ):
        self.alpha[i] += self.y[j] * self.y[i] * (alphaJold - self.alpha[j])
        
        
        
    def saturate_alpha_j(self, j, bounds):
        if self.alpha[j] < bounds[0]: 
            self.alpha[j] = bounds[0]            
        if self.alpha[j] > bounds[1]: 
            self.alpha[j] = bounds[1]
    
    
    
    def check_if_alpha_violates_kkt(self, alpha, E):
        return (alpha > 0 and np.abs(E) < EPS) or (alpha < C and np.abs(E) > EPS)
    
    
 # set which bounds we must assign to alpha ( based on Yi and Yj)   
    def bound_alpha(self, alphai, alphaj, yi, yj):
        bounds = [2]
        if (yi == yj):
            bounds.insert(0, max(0, alphaj + alphai - C))
            bounds.insert(1, min(C, alphaj + alphai))
        else:
            bounds.insert(0, max(0, alphaj - alphai))
            bounds.insert(1, min(C, alphaj - alphai + C))
        return bounds
    
    
    
    def calc_w(self, alpha, x, y):
        w = np.zeros((np.shape(x)[1], 1))
        for i in range (np.shape(x)[0]):
            w += np.multiply(y[i]*alpha[i],x[i].T)
        return w
    
    
    
    def classify(self, x):
        classification = -1
        if (np.sign((np.dot(x, self.w) + self.b).item(0,0)) == 1):
            classification = 1
        return classification
    

### GET METHODS   
    def get_alpha(self):
        return self.alpha    
    def get_x(self):
        return self.x    
    def get_b(self):
        return self.b   
    def get_w(self):
        return self.w    
    def get_positions(self):
        return self.positions
    def get_support_vectors(self):
        return self.support_vectors 
    def get_support_vectors_y(self):
        return self.support_vectors_y     
#############################################    
    

def plot (X1, X2, X, alpha, b, w):
    x = np.arange(-2.0 , 5.0, 0.1)
    y = (-w[0] * x - b) / w[1]
    pos = svm.get_positions()
    yplus = (-w[0] * x - b + 1 ) / w[1]
    yminus = (-w[0] * x - b - 1 ) / w[1]
    plt.scatter(X1[:,0], X1[:,1], c='g')
    plt.scatter(X2[:,0], X2[:,1])
    plt.scatter(X[pos,0], X[pos,1], s=100, c='m',marker="*")
    plt.plot(x,y, c='k')
    plt.plot(x,yplus, 'k--', c='r')
    plt.plot(x,yminus, 'k--', c='r')
    
    
        
def test():
   Xtest = [0,0]
   Xtest[0] = float(input("enter X coordinate of a point to classify: "))
   Xtest[1] = float(input("enter Y coordinate of a point to classify: "))
   Ytest = svm.classify(Xtest)
   if Ytest == 1:
       print('classification: 1 (over hyperplane)')
   else:
       print('classification: -1 (under hyperplane)')
   plt.scatter(Xtest[0], Xtest[1], c='r')
   
   

################################################################
####  
##############################################################

X1, y1, X2, y2 = create_data()


plt.scatter(X1[:,0], X1[:,1], c='g')
plt.scatter(X2[:,0], X2[:,1])
plt.grid()
plt.show()


X = np.vstack((X1, X2))
y = np.hstack((y1, y2))


xArray=[]
yArray=[]

for i in range(len(X)):
    xArray.append(X[i])
    yArray.append(y[i])
    
svm = SVM(np.mat(xArray), np.mat(yArray).transpose())

plot(X1, X2, X, svm.get_alpha(), svm.get_b().item(0,0), svm.get_w())  
test()

plt.grid()
plt.show()

















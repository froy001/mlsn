import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
    posx = []
    posy = []
    negx = []
    negy = []
    for i in range(len(y)):
        if y[i] == 1:
            posx.append(X[i][0])
            posy.append(X[i][1])
        else:
            negx.append(X[i][0])
            negy.append(X[i][1])
    plt.plot(posx,posy,'k*',label='y=1')
    plt.plot(negx,negy,'yo',label='y=0')
    plt.ylabel('Microchip Test 2')
    plt.xlabel('Microchip Test 1')
    plt.legend()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def mapFeature(X1,X2,degree):
    out = np.matrix(np.ones(X1.shape[0])).T
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.c_[out,np.multiply(np.power(X1,i-j),np.power(X2,j))]
    return out
    
def costReg(theta,X,y,lamb):
    XbyT = X*theta
    return ((-y.T*np.log(sigmoid(XbyT))-(1-y).T*np.log(1-sigmoid(XbyT)))+theta[1:,:].T*theta[1:,:]*lamb/2)/y.shape[0]

def gradientReg(theta,X,y,lamb):
    sigXbyT = sigmoid(X*theta)
    grad = np.matrix(np.zeros(theta.shape[0])).T
    grad[0] = X[:,0].T*(sigXbyT-y)
    grad[1:] = X[:,1:].T*(sigmoid(X*theta)-y)+lamb*theta[1:]
    return grad/y.shape[0]
    
def gdbt(theta,X,y,lamb,alpha,beta,iter,tol):
    # This gdbt is different from that in courseraEx01.py, since the cost and gradient are different
    # gdbt: Gradient Descent BackTracking
    # Gradient descent minimization based on backtracking line search
    # Output: Minimizer within precision or maximum iteration number
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iter: Maximum number of iterations
        # tol: The procedure will break if the square of the 2-norm of the gradient is less than the threshold tol
    for i in range(iter):
        grad = gradientReg(theta,X,y,lamb)
        delta = -grad
        if grad.T*grad < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta
        J = costReg(theta,X,y,lamb)
        alphaGradDelta = alpha*grad.T*delta
        # begin line search
        t = 1
        while costReg(theta+t*delta,X,y,lamb) > J+t*alphaGradDelta:
            t = beta*t
        # end line search

        # update
        theta = theta+t*delta
    return theta

def plotDecisionBoundary(X,degree):
    x1Array = np.linspace(np.min(X[:,1]),np.max(X[:,1]),num=50)
    x2Array = np.linspace(np.min(X[:,2]),np.max(X[:,2]),num=50)
    z = np.zeros((len(x1Array),len(x2Array)))
    for i in range(len(x1Array)):
        for j in range(len(x2Array)):
            z[i,j] = mapFeature(np.matrix(x1Array[i]),np.matrix(x2Array[j]),degree)*theta
    plt.contour(x1Array,x2Array,z,1)
    
def predict(theta,X):
    return np.matrix([1 if sigmoid(x*theta)>=0.5 else 0 for x in X]).T
    
# Load data
f = open('ex2data2.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append([float(data[0]),float(data[1])])
    y.append(float(data[2]))

plotData(X,y)

X = np.matrix(X)
y = np.matrix(y).T

degree = 6

# Part 1: Regularized logistic regression
X = mapFeature(X[:,0],X[:,1],degree)

initial_theta = np.matrix(np.zeros(X.shape[1])).T

cost = costReg(initial_theta,X,y,1)
gradient = gradientReg(initial_theta,X,y,1)

print 'Cost at initial theta (zeros):',cost

# Part 2: Regularization and accuracies

lamb = 10

alpha = 0.01
beta = 0.8

theta = gdbt(initial_theta,X,y,lamb,alpha,beta,1000,1e-8)

plotDecisionBoundary(X,degree)

p = predict(theta,X)

print 'Train accuracy:',(len(y)-np.sum(np.abs(y-p)))/len(y)*100,'%'

plt.savefig('courseraEx02_fig02.png')
plt.show()


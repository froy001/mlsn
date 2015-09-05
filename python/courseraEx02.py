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
    plt.plot(posx,posy,'k*',label='Admitted')
    plt.plot(negx,negy,'yo',label='Not admitted')
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    plt.legend()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(theta,X,y):
    XbyT = X*theta
    return (-y.T*np.log(sigmoid(XbyT))-(1-y).T*np.log(1-sigmoid(XbyT)))/y.shape[0]

def gradient(theta,X,y):
    return X.T*(sigmoid(X*theta)-y)/y.shape[0]
    
def gdbt(theta,X,y,alpha,beta,iter,tol):
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
        grad = gradient(theta,X,y)
        delta = -grad
        if grad.T*grad < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta
        J = cost(theta,X,y)
        alphaGradDelta = alpha*grad.T*delta
        # begin line search
        t = 1
        while cost(theta+t*delta,X,y) > J+t*alphaGradDelta:
            t = beta*t
        # end line search

        # update
        theta = theta+t*delta
    return theta
    
def featureNormalize(X): # copied from courseraEx01_multi.py
    mu = np.mean(X,axis=0)
    # numpy std is different from matlab std. The difference will get smaller when huge dataset is processed and the prediction will not be influenced
    sigma = np.std(X,axis=0)
    X_norm = np.divide(X-mu,sigma)
    return (X_norm,mu,sigma)

def predict(theta,X):
    return np.matrix([1 if sigmoid(x*theta)>=0.5 else 0 for x in X]).T
    
# Part 1: Plotting
f = open('../ex2data1.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append([float(data[0]),float(data[1])])
    y.append(float(data[2]))

plotData(X,y)

X = np.matrix(X)
y = np.matrix(y).T

normalized = featureNormalize(X)
X_norm = normalized[0]
mu = normalized[1]
sigma = normalized[2]


# Part 2: Compute cost and gradient
X = np.c_[np.ones(X.shape[0]),X]
X_norm = np.c_[np.ones(X_norm.shape[0]),X_norm]
initial_theta = np.matrix(np.zeros(X.shape[1])).T

costZero = cost(initial_theta,X,y)
gradientZero = gradient(initial_theta,X,y)

print 'Cost at initial theta (zeros):',costZero
print 'Gradient at initial theta (zeros):'
print gradientZero

# Part 3: 
alpha = 0.01
beta = 0.8

thetaUnnorm = gdbt(initial_theta,X_norm,y,alpha,beta,1000,1e-8)
# To get the unnormalized theta
theta = thetaUnnorm
# We can also include those procedures in the function gdbt
theta[0] -= sum(np.multiply(thetaUnnorm[1:],mu.T)/sigma.T)
for i in range(1,len(theta)):
    theta[i] = thetaUnnorm[i]/sigma[0,i-1]

p = predict(theta,X)
print 'Cost at theta found by gdbt:',cost(theta,X,y)
print 'theta:'
print theta

score1 = np.squeeze(np.asarray(np.matrix(np.linspace(30,100,num=200))))
score2 = np.squeeze(np.asarray((-theta[0]-np.multiply(score1,theta[1]))/theta[2]))

plt.plot(score1,score2)
plt.savefig('courseraEx02_fig01.png')

print 'For a student with scores 45 and 85, we predict an admission',sigmoid(np.matrix([1,45,85])*theta)

print 'Train accuracy:',len(y)-np.sum(np.abs(y-p))

plt.show()


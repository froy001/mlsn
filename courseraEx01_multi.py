import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    # numpy std is different from matlab std. The difference will get smaller when huge dataset is processed and the prediction will not be influenced
    sigma = np.std(X,axis=0)
    X_norm = np.divide(X-mu,sigma)
    return (X_norm,mu,sigma)

def computeCostMulti(X,y,theta):
    return np.sum(np.power(X*theta-y,2))/2/len(y)

def gradientDescentMulti(X,y,theta,alpha,num_iters):
    J_history = []
    for i in range(num_iters):
        thetaTemp = np.zeros((len(theta),1))
        for j in range(len(theta)):
            thetaTemp[j] = theta[j] + alpha*np.dot((y-X*theta).T,X[:,j])/len(y)
        theta = thetaTemp
        J_history.append(computeCostMulti(X,y,theta))
    return (theta,J_history)

# Part 1: Feature Normalization
f = open('ex1data2.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append([float(data[0]),float(data[1])])
    y.append(float(data[2]))

X = np.matrix(X)
y = np.matrix([[item] for item in y])

a1 = featureNormalize(X)

X = a1[0]
mu = a1[1]
sigma = a1[2]

X = np.c_[np.ones(len(X)),X]

# Part 2: Gradient descent

alpha = 1
num_iters = 200

theta = np.zeros((3,1))

a2 = gradientDescentMulti(X,y,theta,alpha,num_iters)

theta = a2[0]
J_history = a2[1]

# theta will be different from that calculated by matlab, because we having "different" training data due to the thse std function. By we'll see, the predicted price will be the same.

print 'Theta computed from gradient descent:'
print theta

plt.plot(J_history)
plt.ylabel('Cost over iteration')
plt.xlabel('Number of iteration')
plt.savefig('courseraEx01_multi_fig01.png')

ar = 1650
br = 3

price = np.c_[np.ones(1),(np.matrix([ar,br])-mu)/sigma]*theta

print 'Predicted price of a',ar,'sq-ft,',br,'house (using gradient descent):'
print price

plt.show()


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Python solution to ML open course (Coursera, Andrew Ng)
def computeCost(X,y,theta):
    return np.sum(np.power(X*theta-y,2))/2/len(y)
    
def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        theta_temp0 = theta[0]-alpha*np.sum(X*theta-y)/len(y)
        theta_temp1 = theta[1]-alpha*np.dot(np.transpose(X*theta-y),X[:,1])/len(y)
        theta = np.concatenate((theta_temp0,theta_temp1),axis=0)
        J_history.append(computeCost(X,y,theta))
    return (theta, J_history)

# Part 1: Basic function (skipped, since it's boring)

# Part 2: Plotting
f = open('../ex1data1.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append(float(data[0]))
    y.append(float(data[1]))

plt.figure(1)
plt.plot(X,y,'.',label='Training data')
plt.ylabel('y')
plt.xlabel('x')

# Part 3: Gradient descent
X = np.matrix([[1, item] for item in X])
y = np.matrix([[item] for item in y])
theta = np.matrix([[0],[0]])
iterations = 1500
alpha = 0.01;

print 'Test computeCost function: cost = ', computeCost(X,y,theta)

num_iters = 1500
a = gradientDescent(X, y, theta, alpha, num_iters)

theta = a[0]
J_history = a[1]
print 'Theta found by gradient descent:', theta.tolist()

plt.figure(2)
plt.plot(J_history)
plt.ylabel('Cost over iteration')
plt.xlabel('Number of iteration')
plt.savefig('courseraEx01_fig01.png')

plt.figure(1)
XbyTheta = X*theta
plt.plot(X[:,1].tolist(),XbyTheta.tolist(),'r',label='Linear regression')
plt.legend()
plt.savefig('courseraEx01_fig02.png')

predict1 = np.matrix([1,3.5])*theta
print 'For population = 35,000, we predict a profit of', predict1.tolist()[0][0]*10000
predict2 = np.matrix([1,7])*theta
print 'For population = 70,000, we predict a profit of', predict2.tolist()[0][0]*10000

# Part 4: Visualizing J(theta_0, theta_1)

theta0_vals = np.linspace(-10,10,num=100)
theta1_vals = np.linspace(-1,4,num=100)

J_vals = [[computeCost(X,y,np.matrix([[t0],[t1]]))for t0 in theta0_vals] for t1 in theta1_vals]

X, Y = np.meshgrid(theta0_vals,theta1_vals)

fig = plt.figure(3)
ax = fig.gca(projection='3d')
plt.savefig('courseraEx01_fig03.png')

# 3-D surface
surf = ax.plot_surface(X, Y, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# contour
plt.figure(4)
plt.contour(X,Y,J_vals,100)
plt.plot(float(theta[0]),float(theta[1]),'rx')
plt.savefig('courseraEx01_fig04.png')

plt.show()

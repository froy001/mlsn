from funcsEx05 import *

# Part 1: Loading and visualizing data
mat_contents = sio.loadmat('../ex5data1.mat')
X = np.matrix(mat_contents['X'])
Xtest = np.matrix(mat_contents['Xtest'])
Xval = np.matrix(mat_contents['Xval'])
y = np.matrix(mat_contents['y'])
ytest = np.matrix(mat_contents['ytest'])
yval = np.matrix(mat_contents['yval'])

plt.plot(X,y,'rx',markersize=10)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression fit (lambda = 0)')

# Part 2 and 3: Regularized linear regression cost and gradient
theta = np.matrix(np.array([1,1])).T
J,grad = linearRegCostFunction(np.c_[np.ones((y.shape[0],1)),X],y,theta,1)

print 'Cost at theta =',theta.tolist(),':',J.tolist()[0][0],'\n'
print 'Gradient at theta =',theta.tolist(),':',grad.tolist(),'\n'


# Part 4: Train linear regression
lamb = 1
theta = trainLinearRegBT(np.c_[np.ones((y.shape[0],1)),X],y,lamb,0.25,0.5,50,1e-6)
plt.plot(X,np.c_[np.ones((y.shape[0],1)),X]*theta,'-',linewidth=2)
plt.savefig('courseraEx05_fig01.png')
plt.show()


# Part 5: Learning curve fur linear regression
lamb = 0
error_train,error_val = learningCurve(np.c_[np.ones((y.shape[0],1)),X],y,np.c_[np.ones((yval.shape[0],1)),Xval],yval,lamb,50)
plt.plot(np.linspace(1,y.shape[0]+1,y.shape[0]),error_train,label='Train')
plt.plot(np.linspace(1,y.shape[0]+1,y.shape[0]),error_val,label='Cross validation')
plt.axis([0,12,0,150])
plt.ylabel('Error')
plt.xlabel('Number of training examples')
plt.legend()
plt.title('Polynomial regression learning curve (lambda=0)')

plt.savefig('courseraEx05_fig02.png')
plt.show()
print '# Training examples   Train error        Cross validation error'
for i in range(y.shape[0]):
    print i+1,'                  ',error_train[i],'     ',error_val[i]


# Part 6: Feature mapping for polynomial regression
p = 8
X_poly = polyFeatures(X,p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.c_[np.ones(X_poly.shape[0]),X_poly]

X_poly_val = polyFeatures(Xval,p)
X_poly_val = X_poly_val-mu
X_poly_val = X_poly_val/sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]),X_poly_val]


# Part 7: Learning curve for polynomial regression
lamb = 0.0
theta = trainLinearRegBT(X_poly,y,lamb,0.25,0.5,200,1e-6)

plt.plot(X,y,'rx',markersize=10,linewidth=2)
x = np.matrix(np.arange(X.min()-15,X.max()+15,0.05)).T
X_poly_temp = np.c_[np.ones(x.shape[0]),(polyFeatures(x,p)-mu)/sigma]
plt.plot(x,X_poly_temp*theta,'--',linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression fit (lambda = 0)')
plt.savefig('courseraEx05_fig03.png')
plt.show()

error_train,error_val = learningCurve(X_poly,y,X_poly_val,yval,lamb,200)
print '# Training examples        Train error        Cross validation error'
for i in range(y.shape[0]):
    print i+1,'       ',error_train[i][0],'     ',error_val[i][0]

plt.plot(np.linspace(1,y.shape[0]+1,y.shape[0]),error_train,label='Train')
plt.plot(np.linspace(1,y.shape[0]+1,y.shape[0]),error_val,label='Cross validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Polynomial regression learning curve (lambda=0)')
plt.axis([0,13,-10,100])
plt.savefig('courseraEx05_fig04.png')
plt.show()


# Part 8: Validation for selecting lambda
lambda_vec = np.matrix([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
error_train,error_val = validationCurve(X_poly,y,X_poly_val,yval,lambda_vec,50)
print 'lambda          Train error        Cross validation error'
for i in range(lambda_vec.shape[0]):
    print lambda_vec[i].tolist()[0][0],'        ',error_train[i][0],'     ',error_val[i][0]

plt.plot(lambda_vec,error_train,label='Train')
plt.plot(lambda_vec,error_val,label='Cross validation')
plt.ylabel('Error')
plt.xlabel('lambda')
plt.legend()
plt.axis([0,10,0,20])
plt.savefig('courseraEx05_fig05.png')
plt.show()



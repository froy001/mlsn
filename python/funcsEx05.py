import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys

def linearRegCostFunction(X,y,theta,lamb):
    XtMinusY = X*theta-y
    J = np.sum(np.power(XtMinusY,2)) + lamb*(theta[1:].T*theta[1:])
    grad = X.T*XtMinusY
    grad[1:] += lamb*theta[1:]
    return J/2/y.shape[0],np.matrix(grad)/y.shape[0]


def linearRegCost(X,y,theta,lamb):
    J = np.sum(np.power(X*theta-y,2)) + lamb*(theta[1:].T*theta[1:])
    return J/2/y.shape[0]


def linearRegGradient(X,y,theta,lamb):
    grad = X.T*(X*theta-y)
    grad[1:] += theta[1:]*lamb
    return np.matrix(grad)/y.shape[0]


def linearRegHessian(X,theta,lamb):
    Htheta = np.matrix(np.eye(X.shape[1]))
    Htheta[0][0] = 0
    return (X.T*X + np.multiply(lamb,Htheta))/X.shape[0]


def trainLinearRegBT(X,y,lamb,alpha,beta,iter,tol):
    theta = np.matrix(np.zeros((X.shape[1],1)))

    for i in range(iter):
        # Gradient
        grad = linearRegGradient(X,y,theta,lamb)
        # Hessian
        H = linearRegHessian(X,theta,lamb)
        # Newton step
        if np.linalg.cond(H) > 1/sys.float_info.epsilon:
            delta = np.linalg.solve(np.diag(np.diag(H)),-grad)
        else:
            delta = np.linalg.solve(H,-grad)

        if grad.T*grad < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta

        J = linearRegCost(X,y,theta,lamb)
        alphaGradDelta = alpha*grad.T*delta
        t = 1
        Jnew = linearRegCost(X,y,theta+t*delta,lamb)
        # begin line search
        while Jnew > J+t*alphaGradDelta or np.isnan(Jnew):
            t = beta*t
            Jnew = linearRegCost(X,y,theta+t*delta,lamb)
        # end line search
        # update

        theta = theta+t*delta
        print 'Iteration',i+1,' | Cost:',Jnew
    return theta
    
        
def learningCurve(X,y,Xval,yval,lamb,iter):
    error_train = np.zeros((y.shape[0],1))
    error_val = np.zeros((y.shape[0],1))
    for i in range(y.shape[0]):
        theta = trainLinearRegBT(X[:i+1,:],y[:i+1],lamb,0.25,0.5,iter,1e-12)
        error_train[i] = linearRegCost(X[:i+1,:],y[:i+1],theta,0)
        error_val[i] = linearRegCost(Xval,yval,theta,0)
    return error_train,error_val


def validationCurve(X,y,Xval,yval,lamb_vec,iter):
    error_train = np.zeros((lamb_vec.shape[0],1))
    error_val = np.zeros((lamb_vec.shape[0],1))
    print X,Xval
    for i in range(lamb_vec.shape[0]):
        lamb = lamb_vec[i][0]
        theta = trainLinearRegBT(X,y,lamb,0.25,0.5,iter,1e-12)
        error_train[i] = linearRegCost(X,y,theta,0)
        error_val[i] = linearRegCost(Xval,yval,theta,0)
    return error_train,error_val


def polyFeatures(X,p):
    # X is a vector / column
    X_poly = np.matrix(np.zeros((X.shape[0],p)))
    X_poly[:,0] = X
    for i in range(1,p):
        X_poly[:,i] = np.multiply(X_poly[:,i-1],X)
    return X_poly


def featureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0,ddof=1)
    X_norm = (X-mu)/sigma
    return X_norm,mu,sigma


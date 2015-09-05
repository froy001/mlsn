import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def displayData(X):
    # python translation of displayData.m from coursera
    # For now, only "quadratic" image
    example_width = np.round(np.sqrt(X.shape[1]))
    example_height = example_width
    
    display_rows = np.floor(np.sqrt(X.shape[0]))
    display_cols = np.ceil(X.shape[0]/display_rows)

    pad = 1

    display_array = -np.ones((pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))

    curr_ex = 0

    for j in range(display_rows.astype(np.int16)):
        for i in range(display_cols.astype(np.int16)):
            if curr_ex == X.shape[0]:
                break
            max_val = np.max(np.abs(X[curr_ex,:]))
            rowStart = pad+j*(example_height+pad)
            colStart = pad+i*(example_width+pad)
            display_array[rowStart:rowStart+example_height, colStart:colStart+example_width] = X[curr_ex,:].reshape((example_height,example_width)).T/max_val

            curr_ex += 1
        if curr_ex == X.shape[0]:
            break

    plt.imshow(display_array,extent = [0,10,0,10],cmap = plt.cm.Greys_r)
    plt.savefig('courseraEx03_fig01.png')
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costReg(theta,X,y,lamb):
    XbyT = X*theta
    return ((-y.T*np.log(sigmoid(XbyT))-(1-y).T*np.log(1-sigmoid(XbyT)))+theta[1:,:].T*theta[1:,:]*lamb/2)/y.shape[0]

def gradientReg(theta,X,y,lamb):
    sigXbyT = sigmoid(X*theta)
    grad = np.matrix(np.zeros(theta.shape[0])).T
    grad[0] = X[:,0].T*(sigXbyT-y)
    grad[1:] = X[:,1:].T*(sigmoid(X*theta)-y)+lamb*theta[1:]
    return grad/y.shape[0]

def hessianReg(theta,X,lamb):
    sigXbyT = sigmoid(X*theta)
    H = X.T*np.diag(np.sqrt(np.multiply(sigXbyT,1-sigXbyT)))
    Htheta = np.matrix(np.eye(X.shape[1]))
    Htheta[0][0] = 0
    return (np.dot(H,H.T)+lamb*Htheta)/X.shape[0]

def newtonbt(theta,X,y,lamb,alpha,beta,iter,tol):
    # newtonbt: Newton's method with backtracking line search
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iter: Maximum number of iterations
        # tol: The procedure will break if the square of the Newton decrement is less than the threshold tol
    for i in range(iter):
        # Gradient
        grad = gradientReg(theta,X,y,lamb)
        # Hessian
        H = hessianReg(theta,X,lamb)
        # Newton step
        xnt = np.linalg.solve(H,-grad)
        # Newton decrement (squared)
        ntdec = -grad.T*xnt
        if abs(ntdec)/2 < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta
        J = costReg(theta,X,y,lamb)
        alphaGradx = alpha*(grad.T*xnt)
        # begin line search
        t = 1
        costNew = costReg(theta+t*xnt,X,y,lamb)
        while costNew > J+t*alphaGradx or np.isnan(costNew):
            t = beta*t
            costNew = costReg(theta+t*xnt,X,y,lamb)
        # end line search

        # update
        theta += t*xnt
    print 'cost:',J
    return theta
        
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
    print 'Cost:',costReg(theta,X,y,lamb)
    return theta

def oneVsAll(X,y,num_labels,lamb):
    X = np.c_[np.ones((X.shape[0],1)),X]
    all_theta = np.zeros((num_labels,X.shape[1]))
    for i in range(num_labels):
        initial_theta = np.matrix(np.zeros(X.shape[1])).T
        yidx = np.matrix([1 if y[j] == i+1 else 0 for j in range(X.shape[0])]).T
        # Gradient descent method
#        all_theta[i,:] = gdbt(initial_theta,X,yidx,lamb,0.25,0.5,100,1e-8).T
        # Newton's method
        all_theta[i,:] = newtonbt(initial_theta,X,yidx,lamb,0.25,0.5,50,1e-8).T
    return all_theta

def predictOneVsAll(all_theta,X):
    X = np.c_[np.ones((X.shape[0],1)),X]
    h = np.dot(X,all_theta.T)
    y = np.matrix(np.zeros(X.shape[0])).T
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if h[i,j] == np.max(h[i,:]):
                y[i] = j+1
    return y

input_layer_size  = 400
num_labels = 10

# Part 1: Loading and visualizing data

mat_contents = sio.loadmat('../ex3data1.mat')
X = mat_contents['X']
y = mat_contents['y']

rand_indices = np.random.permutation(X.shape[0])
sel = X[rand_indices[0:100],:]

displayData(sel)

# Part 2: Vectorize logistic regression
lamb = 0.1
all_theta = oneVsAll(X,y,num_labels,lamb)

pred = predictOneVsAll(all_theta,X)

precision = 0
for i in range(len(y)):
    if y[i] == pred[i]:
        precision += 1

print 'Training Set Accuracy:',(1.0*precision)/len(y)





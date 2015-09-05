import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.axes as ax

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

def predict(Theta1,Theta2,X):
    X = np.c_[np.ones((X.shape[0],1)),X]
    z2 = np.dot(Theta1,X.T)
    a2 = np.r_[np.ones((1,z2.shape[1])),sigmoid(z2)]
    z3 = np.dot(Theta2,a2)
    a3 = sigmoid(z3)
    p = np.zeros((X.shape[0],1))
    for i in range(a3.shape[1]):
        for j in range(a3.shape[0]):
            if a3[j,i] == np.max(a3[:,i]):
                p[i] = j+1
    return p
            

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

# Part 1: Loading and visualizing data

mat_contents = sio.loadmat('../ex3data1.mat')
X = mat_contents['X']
y = mat_contents['y']

rand_indices = np.random.permutation(X.shape[0])
sel = X[rand_indices[0:100],:]

displayData(sel)

# Part 2: Loading parameter

mat_contents = sio.loadmat('../ex3weights.mat')

Theta1 = mat_contents['Theta1']
Theta2 = mat_contents['Theta2']

# Part 3: Implement predict

p = predict(Theta1,Theta2,X)

precision = 0
for i in range(len(y)):
    if y[i] == p[i]:
        precision += 1

print 'Training Set Accuracy:',(1.0*precision)/len(y)

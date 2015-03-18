from funcsEx04 import *

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

# Part 1: Loading and visualizing data

mat_contents = sio.loadmat('ex4data1.mat')
X = mat_contents['X']
y = mat_contents['y']

rand_indices = np.random.permutation(X.shape[0])
sel = X[rand_indices[0:100],:]

displayData(sel,'courseraEx04_fig01.png')

# Part 2: Loading parameters

mat_contents = sio.loadmat('ex4weights.mat')

Theta1 = mat_contents['Theta1']
Theta2 = mat_contents['Theta2']

nn_params1 = np.matrix(np.reshape(Theta1, Theta1.shape[0]*Theta1.shape[1], order='F')).T
nn_params2 = np.matrix(np.reshape(Theta2, Theta2.shape[0]*Theta2.shape[1], order='F')).T
nn_params = np.r_[nn_params1,nn_params2]

# Part 3: Compute cost (feedforward)

lamb = 0

J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)

print 'Cost at parameters without regulariyation (loaded from ex4weights):',J,'\n'

# Part 4: Implement regularization
lamb = 1

J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)

print 'Cost at parameters with regularization (loaded from ex4weights):',J,'\n'

# Part 5: Sigmoid gradient
g = sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))

print 'Evaluating sigmoid gradient\n',g,'\n'

# Part 6: Initializing parameters

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)

initial_nn_params = np.r_[np.reshape(initial_Theta1,Theta1.shape[0]*Theta1.shape[1],order='F'),np.reshape(initial_Theta2,Theta2.shape[0]*Theta2.shape[1],order='F')]

# Part 7: Implement backpropagation
checkNNGradients(0)

# Part 8: Implement regularization
lamb = 3
checkNNGradients(lamb)
debug_J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
print 'Cost at (fixed) debugging parameters (lambda =',lamb,'):',debug_J[0],'\n'

# Part 9: Training NN
lamb = 1
Theta = cgbt(initial_nn_params,X,y,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,500,1e-8)

Theta1 = np.matrix(np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
Theta2 = np.matrix(np.reshape(Theta[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))

# Part 10: Visualize weights
displayData(Theta1[:,1:],'courseraEx04_fig02.png')

# Part 11: Implement predict
p = predict(Theta1,Theta2,X)

precision = 0
for i in range(len(y)):
    if y[i] == p[i]:
        precision += 1

print 'Training Set Accuracy:',(1.0*precision)/len(y)

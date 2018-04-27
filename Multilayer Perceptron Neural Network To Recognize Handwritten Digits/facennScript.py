'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
from scipy.optimize import minimize
import numpy as np
import pickle
from math import sqrt


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix

    # return the sigmoid of input z"""

    z = np.asarray(z)

    return  1 / (1 + np.exp(-z))



# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):

    """% nnObjFunction computes the value of objective function (negative log 

    %   likelihood error function with regularization) given the parameters 

    %   of Neural Networks, thetraining data, their corresponding training 

    %   labels and lambda - regularization hyper-parameter.



    % Input:

    % params: vector of weights of 2 matrices w1 (weights of connections from

    %     input layer to hidden layer) and w2 (weights of connections from

    %     hidden layer to output layer) where all of the weights are contained

    %     in a single vector.

    % n_input: number of node in input layer (not include the bias node)

    % n_hidden: number of node in hidden layer (not include the bias node)

    % n_class: number of node in output layer (number of classes in

    %     classification problem

    % training_data: matrix of training data. Each row of this matrix

    %     represents the feature vector of a particular image

    % training_label: the vector of truth label of training images. Each entry

    %     in the vector represents the truth label of its corresponding image.

    % lambda: regularization hyper-parameter. This value is used for fixing the

    %     overfitting problem.

       

    % Output: 

    % obj_val: a scalar value representing value of error function

    % obj_grad: a SINGLE vector of gradient value of error function

    % NOTE: how to compute obj_grad

    % Use backpropagation algorithm to compute the gradient of error function

    % for each weights in weight matrices.



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % reshape 'params' vector into 2 matrices of weight w1 and w2

    % w1: matrix of weights of connections from input layer to hidden layers.

    %     w1(i, j) represents the weight of connection from unit j in input 

    %     layer to unit i in hidden layer.

    % w2: matrix of weights of connections from hidden layer to output layers.

    %     w2(i, j) represents the weight of connection from unit j in hidden 

    %     layer to unit i in output layer."""



    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args



    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))

    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0



    # Your code here. Start -->
    
    n = train_data.shape[0]
    f_len = train_data.shape[1]
    
    train_data = np.append(train_data, np.zeros(shape=(n,1)) + 1, axis=1)

    a = np.dot(w1, train_data.T) #a_j
    #(50, 4996)
    
    z = sigmoid(a) #z_j
    #(50, 4996)
    
    z = z.T
    #(4996, 50)
    
    b = np.dot(w2, np.append(z, np.zeros(shape=(n, 1)) + 1, axis=1).T) #b_l
    #(10, 4996)
    
    o = sigmoid(b) #o_l
    #(10, 4996)
    
    y = np.zeros(shape=(n_class, n))
    y = np.double(y)
    #(10, 4996)
    
    for i in range(0, n): #1-of-K transformation
        y[train_label[i].astype(int)][i] = 1.0
        
    #Error function value
    obj_val = (-1/n) * np.sum(np.sum((y * np.log(o) + (1 - y) * np.log(1 - o)), axis=0)) + (lambdaval / (2*n)) * (np.sum(w1**2) + np.sum(w2**2))
    print('\n' + str(obj_val))
    

    #grad_w2
    del_l = o - y
    w2_temp = np.empty((n_class,n_hidden+1,n))
    w2_temp = np.double(w2_temp)
    for i in range(0, n):
        w2_temp[:,:,i] = np.repeat(np.reshape(np.append(z[i], [1]), (1, n_hidden+1)), n_class, axis=0) * np.reshape(del_l[:,i], (n_class,1))  
    grad_w2 = (np.sum(w2_temp, axis=2) + (lambdaval * w2)) / n
    #(10, 51)
    
    #grad_w1
    w1_temp = np.empty((n_hidden,f_len+1,n))
    grad_sum = np.empty((n_class,n_hidden+1,n))
    w1_temp = np.double(w1_temp)
    grad_sum = np.double(grad_sum)
    for i in range(0, n):
        grad_sum[:,:,i] = w2 * np.reshape(del_l[:,i], (n_class,1))
    del_sum_term = np.sum(grad_sum, axis=0)
    del_sum_term = del_sum_term[:n_hidden,:]
    for i in range(0, n):
        w1_temp[:,:,i] = np.repeat(np.reshape(train_data[i], (1, f_len+1)), n_hidden, axis=0) * np.reshape(z[i,:] * (1 - z[i,:]), (n_hidden,1)) * np.reshape(del_sum_term[:,i], (n_hidden,1))
    grad_w1 = (np.sum(w1_temp, axis=2) + (lambdaval * w1)) / n
    #(50, 530)
    
    # Your code here. <-- End




    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2

    # you would use code similar to the one below to create a flat array

    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)


    return (obj_val, obj_grad)



    
# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural

    % Network.



    % Input:

    % w1: matrix of weights of connections from input layer to hidden layers.

    %     w1(i, j) represents the weight of connection from unit i in input 

    %     layer to unit j in hidden layer.

    % w2: matrix of weights of connections from hidden layer to output layers.

    %     w2(i, j) represents the weight of connection from unit i in input 

    %     layer to unit j in hidden layer.

    % data: matrix of data. Each row of this matrix represents the feature 

    %       vector of a particular image

       

    % Output: 

    % label: a column vector of predicted labels"""



    labels = np.array([])

    # Your code here. Start -->
    
    data = np.append(data, np.zeros(shape=(data.shape[0],1)) + 1, axis=1)

    a = np.dot(w1, data.T) #a_j
    #(50, 4996)
    
    z = sigmoid(a) #z_j
    #(50, 4996)
    
    z = z.T
    #(4996, 50)
    
    b = np.dot(w2, np.append(z, np.zeros(shape=(data.shape[0], 1)) + 1, axis=1).T) #b_l
    #(10, 4996)
    
    o = sigmoid(b) #o_l
    #(10, 4996)
    
    dec = np.array([0,1])
    dec = np.reshape(np.double(dec),(1,2))
    kTo1 = np.dot(dec, (o / np.amax(o, axis=0)).astype(int))
    labels = np.reshape(kTo1, (data.shape[0],))

    # Your code here. Ends <--

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
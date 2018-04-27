import numpy as np

from scipy.optimize import minimize

from scipy.io import loadmat

from math import sqrt

import pickle

import datetime


#--> global variable for descent plot
obj_func_desc = np.array([])


def initializeWeights(n_in, n_out):

    """

    # initializeWeights return the random weights for Neural Network given the

    # number of node in the input layer and output layer



    # Input:

    # n_in: number of nodes of the input layer

    # n_out: number of nodes of the output layer

       

    # Output: 

    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""



    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)

    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon

    return W





def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix

    # return the sigmoid of input z"""

    z = np.asarray(z)

    return  1 / (1 + np.exp(-z))





def preprocess_small():

    """ Input:

     Although this function doesn't have any input, you are required to load

     the MNIST data set from file 'mnist_sample.mat'.



     Output:

     train_data: matrix of training set. Each row of train_data contains 

       feature vector of a image

     train_label: vector of label corresponding to each image in the training

       set

     validation_data: matrix of training set. Each row of validation_data 

       contains feature vector of a image

     validation_label: vector of label corresponding to each image in the 

       training set

     test_data: matrix of training set. Each row of test_data contains 

       feature vector of a image

     test_label: vector of label corresponding to each image in the testing

       set



     - feature selection"""





    mat = loadmat('mnist_sample.mat')

        # ------------Initialize preprocess arrays----------------------#

    train_preprocess = np.zeros(shape=(4996, 784))

    validation_preprocess = np.zeros(shape=(1000, 784))

    test_preprocess = np.zeros(shape=(996, 784))

    train_label_preprocess = np.zeros(shape=(4996,))

    validation_label_preprocess = np.zeros(shape=(1000,))

    test_label_preprocess = np.zeros(shape=(996,))

    # ------------Initialize flag variables----------------------#

    train_len = 0

    validation_len = 0

    test_len = 0

    train_label_len = 0

    validation_label_len = 0

    # ------------Start to split the data set into 6 arrays-----------#

    for key in mat:

        # -----------when the set is training set--------------------#

        if "train" in key:

            label = key[-1]  # record the corresponding label

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)  # get the length of current training set

            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set



            # ---------------------adding data to training set-------------------------#

            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]

            train_len += tag_len



            train_label_preprocess[train_label_len:train_label_len + tag_len] = label

            train_label_len += tag_len



            # ---------------------adding data to validation set-------------------------#

            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]

            validation_len += 100



            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label

            validation_label_len += 100



            # ---------------------adding data to test set-------------------------#

        elif "test" in key:

            label = key[-1]

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)

            test_label_preprocess[test_len:test_len + tup_len] = label

            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]

            test_len += tup_len

            # ---------------------Shuffle,double and normalize-------------------------#

    train_size = range(train_preprocess.shape[0])

    train_perm = np.random.permutation(train_size)

    train_data = train_preprocess[train_perm]

    train_data = np.double(train_data)

    train_data = train_data / 255.0

    train_label = train_label_preprocess[train_perm]



    validation_size = range(validation_preprocess.shape[0])

    vali_perm = np.random.permutation(validation_size)

    validation_data = validation_preprocess[vali_perm]

    validation_data = np.double(validation_data)

    validation_data = validation_data / 255.0

    validation_label = validation_label_preprocess[vali_perm]

    

    test_size = range(test_preprocess.shape[0])

    test_perm = np.random.permutation(test_size)

    test_data = test_preprocess[test_perm]

    test_data = np.double(test_data)

    test_data = test_data / 255.0

    test_label = test_label_preprocess[test_perm]



    # Feature selection

    # Your code here. Start -->
    
    train_data_new = np.zeros(shape=(4996, 529))
    
    validation_data_new = np.zeros(shape=(1000, 529))

    test_data_new = np.zeros(shape=(996, 529))
    
    for i in range(0,train_data_new.shape[0]):
        train_data_new[i] = np.reshape(train_data[i],((28,28)))[2:25,2:25].flatten()
        
    for i in range(0,validation_data_new.shape[0]):
        validation_data_new[i] = np.reshape(validation_data[i],((28,28)))[2:25,2:25].flatten()
        
    for i in range(0,test_data_new.shape[0]):
        test_data_new[i] = np.reshape(test_data[i],((28,28)))[2:25,2:25].flatten()
        
    train_data = train_data_new
    
    validation_data = validation_data_new
    
    test_data = test_data_new
    
    # Your code here. <-- End

    print('preprocess_small done')





    return train_data, train_label, validation_data, validation_label, test_data, test_label





def preprocess():

    """ Input:

     Although this function doesn't have any input, you are required to load

     the MNIST data set from file 'mnist_all.mat'.



     Output:

     train_data: matrix of training set. Each row of train_data contains 

       feature vector of a image

     train_label: vector of label corresponding to each image in the training

       set

     validation_data: matrix of training set. Each row of validation_data 

       contains feature vector of a image

     validation_label: vector of label corresponding to each image in the 

       training set

     test_data: matrix of training set. Each row of test_data contains 

       feature vector of a image

     test_label: vector of label corresponding to each image in the testing

       set



     Some suggestions for preprocessing step:

     - feature selection"""



    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary



    # Pick a reasonable size for validation data



    # ------------Initialize preprocess arrays----------------------#

    train_preprocess = np.zeros(shape=(50000, 784))

    validation_preprocess = np.zeros(shape=(10000, 784))

    test_preprocess = np.zeros(shape=(10000, 784))

    train_label_preprocess = np.zeros(shape=(50000,))

    validation_label_preprocess = np.zeros(shape=(10000,))

    test_label_preprocess = np.zeros(shape=(10000,))

    # ------------Initialize flag variables----------------------#

    train_len = 0

    validation_len = 0

    test_len = 0

    train_label_len = 0

    validation_label_len = 0

    # ------------Start to split the data set into 6 arrays-----------#

    for key in mat:

        # -----------when the set is training set--------------------#

        if "train" in key:

            label = key[-1]  # record the corresponding label

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)  # get the length of current training set

            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set



            # ---------------------adding data to training set-------------------------#

            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]

            train_len += tag_len



            train_label_preprocess[train_label_len:train_label_len + tag_len] = label

            train_label_len += tag_len



            # ---------------------adding data to validation set-------------------------#

            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]

            validation_len += 1000



            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label

            validation_label_len += 1000



            # ---------------------adding data to test set-------------------------#

        elif "test" in key:

            label = key[-1]

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)

            test_label_preprocess[test_len:test_len + tup_len] = label

            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]

            test_len += tup_len

            # ---------------------Shuffle,double and normalize-------------------------#

    train_size = range(train_preprocess.shape[0])

    train_perm = np.random.permutation(train_size)

    train_data = train_preprocess[train_perm]

    train_data = np.double(train_data)

    train_data = train_data / 255.0

    train_label = train_label_preprocess[train_perm]



    validation_size = range(validation_preprocess.shape[0])

    vali_perm = np.random.permutation(validation_size)

    validation_data = validation_preprocess[vali_perm]

    validation_data = np.double(validation_data)

    validation_data = validation_data / 255.0

    validation_label = validation_label_preprocess[vali_perm]



    test_size = range(test_preprocess.shape[0])

    test_perm = np.random.permutation(test_size)

    test_data = test_preprocess[test_perm]

    test_data = np.double(test_data)

    test_data = test_data / 255.0

    test_label = test_label_preprocess[test_perm]



    # Feature selection

    # Your code here. Start -->
    
    train_data_new = np.zeros(shape=(50000, 529))
    
    validation_data_new = np.zeros(shape=(10000, 529))

    test_data_new = np.zeros(shape=(10000, 529))
    
    for i in range(0,train_data_new.shape[0]):
        train_data_new[i] = np.reshape(train_data[i],((28,28)))[2:25,2:25].flatten()
        
    for i in range(0,validation_data_new.shape[0]):
        validation_data_new[i] = np.reshape(validation_data[i],((28,28)))[2:25,2:25].flatten()
        
    for i in range(0,test_data_new.shape[0]):
        test_data_new[i] = np.reshape(test_data[i],((28,28)))[2:25,2:25].flatten()
        
    train_data = train_data_new
    
    validation_data = validation_data_new
    
    test_data = test_data_new
    
    # Your code here. <-- End



    print('preprocess done')

    obj_train = [train_data]

    pickle.dump(obj_train, open('selectedfeatures_mnist.pickle', 'wb'))                                             
    return train_data, train_label, validation_data, validation_label, test_data, test_label





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
    
    #Record obj_val in global array for logging and plotting
    global obj_func_desc
    obj_func_desc = np.append(obj_func_desc, obj_val)


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
    
    dec = np.array([0,1,2,3,4,5,6,7,8,9])
    dec = np.reshape(np.double(dec),(1,10))
    kTo1 = np.dot(dec, (o / np.amax(o, axis=0)).astype(int))
    labels = np.reshape(kTo1, (data.shape[0],))

    # Your code here. Ends <--

    return labels





"""**************Neural Network Script Starts here********************************"""



train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#print('\n' + 'n_hidden\t' + 'lambdaval\t' + 'time\t' + 'trs_acc\t' + 'vls_acc\t' + 'tss_acc')

hl_cnt = np.array([4,8,12,16,20,30,50,80])
iter_cnt = np.array([50,100])

for i in hl_cnt:
    
    for j in range(0,81,10):

        for k in iter_cnt:            

            #  Train Neural Network

            obj_func_desc = np.array([])

            # set the number of nodes in input unit (not including bias unit)

            n_input = train_data.shape[1]



            # set the number of nodes in hidden unit (not including bias unit)

            n_hidden = i



            # set the number of nodes in output unit

            n_class = 10



            # initialize the weights into some random matrices

            initial_w1 = initializeWeights(n_input, n_hidden)

            initial_w2 = initializeWeights(n_hidden, n_class)



            # unroll 2 weight matrices into single column vector

            initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)



            # set the regularization hyper-parameter

            lambdaval = j



            args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)



            # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example



            opts = {'maxiter': k}  # Preferred value.



            time_start = datetime.datetime.now()



            nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)



            time_end = datetime.datetime.now()



            time_elapsed = time_end - time_start

            # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal

            # and nnObjGradient. Check documentation for this function before you proceed.

            # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)





            # Reshape nnParams from 1D vector into w1 and w2 matrices

            w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))

            w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


            
            # Write obj to pickle file
            #obj = [train_data, n_hidden, w1, w2, lambdaval]
            obj = [n_hidden, w1, w2, lambdaval]

            pickle.dump(obj, open('params_mnist_'+str(n_hidden)+'_'+str(lambdaval)+'_'+str(k)+'.pickle', 'wb'))



            # Test the computed parameters



            predicted_label = nnPredict(w1, w2, train_data)



            # find the accuracy on Training Dataset



            #print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
            trs_acc = 100 * np.mean((predicted_label == train_label).astype(float))



            predicted_label = nnPredict(w1, w2, validation_data)



            # find the accuracy on Validation Dataset



            #print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
            vls_acc = 100 * np.mean((predicted_label == validation_label).astype(float))


            predicted_label = nnPredict(w1, w2, test_data)



            # find the accuracy on Validation Dataset



            #print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
            tss_acc = 100 * np.mean((predicted_label == test_label).astype(float))


            obj2 = [n_hidden, lambdaval, k, time_elapsed.seconds, trs_acc, vls_acc, tss_acc, obj_func_desc]


            print('\n' + str(n_hidden) + '\t' + str(lambdaval) + '\t' + str(k)  + '\t' + str(time_elapsed.seconds) + '\t' + str(trs_acc) + '%' + '\t' + str(vls_acc) + '%' + '\t' + str(tss_acc) + '%')


            pickle.dump(obj2, open('params_mnist_stats_'+str(n_hidden)+'_'+str(lambdaval)+'_'+str(k)+'.pickle', 'wb'))

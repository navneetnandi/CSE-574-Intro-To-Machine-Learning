import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    N = X.shape[0]      # N = 150
    d = X.shape[1]      # d = 2
    
    unique_classes = np.unique(y)
    unique_classes = unique_classes.reshape(5,1)  #unique_classes = k = 5
    
    means = np.zeros((d, 5))      # A 2x5 Matrix
    
    count0 = 0
    count1 = 0
    sum0 = 0
    sum1 = 0
    
    #Calculating mean of features for every label
    for uc in range(1,5+1):
        count0 = 0
        count1 = 0
        sum0 = 0
        sum1 = 0
        for iter in range(0,150):
            if(y[iter][0] == uc):
                sum0 = sum0 + X[iter][0]
                count0 = count0 + 1
                
                sum1 = sum1 + X[iter][1]
                count1 = count1 + 1
        
        means[0][uc-1] = sum0/count0
        means[1][uc-1] = sum1/count1
    
    
    tr = np.transpose(X)
    covmat = np.cov(tr)
    
    return means,covmat
    
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    N = X.shape[0]      # N = 150
    d = X.shape[1]      # d = 2
    covmats = []
    a = np.empty([1,2])
    
    unique_classes = np.unique(y)
    unique_classes = unique_classes.reshape(5,1)  #unique_classes = k = 5
    
    means = np.zeros((d, 5))      # A 2x5 Matrix
    
    count0 = 0
    count1 = 0
    sum0 = 0
    sum1 = 0
    
    #Calculating mean of features for every label
    for uc in range(1,5+1):
        count0 = 0
        count1 = 0
        sum0 = 0
        sum1 = 0
        cnt=0
        a = np.empty([1,2])
        for iter in range(0,150):
            if(y[iter][0] == uc):
                sum0 = sum0 + X[iter][0]
                count0 = count0 + 1
                
                sum1 = sum1 + X[iter][1]
                count1 = count1 + 1
                
                if(cnt==0):
                    a = X[iter,:]
                    cnt = cnt + 1
                else:
                    a = np.vstack((a, X[iter,:]))
        covmats.append(np.cov(np.transpose(a)))
        
        means[0][uc-1] = sum0/count0
        means[1][uc-1] = sum1/count1      
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    ypred = np.empty([N,1])
    est = np.empty([1,1])
    Xtest = Xtest.T
    acc = 0
    
    for i in range(0,N):
        est = [0]
        for k in range(0,5):
            #calculate the mahanalobis distance
            tmp = [np.dot(np.dot((Xtest[:,i] - means[:,k]).T, inv(covmat)) , (Xtest[:,i] - means[:,k]))]
            est = np.vstack(( est, tmp ))
        
        #find class with lowest value of mahanalobis distance i.e., highest conditional probability
        ypred[i,0] = np.argmin(est[1:,:]) + 1
        
    for i in range(0,N):
        if(ytest[i] - ypred[i] == 0):
            acc = acc + 1
    
    acc = (acc/N)*100
        
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    ypred = np.empty([N,1])
    est = np.empty([1,1])
    Xtest = Xtest.T
    p = np.empty([1,1])
    acc = 0
    
    for i in range(0,N):
        est = [0]
        p = [0]
        for j in range(0,5):
            p = np.vstack((p, np.exp(-0.5 * np.dot(np.dot((Xtest[:,i] - means[:,j]).T, inv(covmats[j])) , (Xtest[:,i] - means[:,j]))) / (2*np.pi*np.power(np.linalg.det(covmats[j]),0.5)) ))
        for k in range(0,5):
            tmp = [p[k+1,:]/np.sum(p)]
            est = np.vstack(( est, tmp ))
        
        #find class with lowest value of mahanalobis distance i.e., highest conditional probability
        ypred[i,0] = np.argmax(est[1:,:]) + 1
        
    for i in range(0,N):
        if(ytest[i] - ypred[i] == 0):
            acc = acc + 1
    
    acc = (acc/N)*100
    
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # IMPLEMENT THIS METHOD  
    
    N = X.shape[0]
    d = X.shape[1]
    
    tr = np.transpose(X)
    inverse = np.linalg.inv(np.dot(tr,X))
    right = np.dot(tr,y)
    w = np.zeros((d,1))
    w = np.dot(inverse,right)
    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    
    N = X.shape[0]
    d = X.shape[1]
    
    w = np.zeros((d,1))
    
    
    idmat = np.identity(d)
    tr = np.transpose(X)
    right = np.dot(tr,y)
    w = np.dot(np.linalg.inv((lambd*idmat) + np.dot(tr,X)), right)
    
    
   # print(left)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    wtr = np.transpose(w)
    loss = 0.0
    
    loss = 0.0
    mse = 0.0
    for i in range(N):
        loss = loss + ((ytest[i][0] - (np.dot(wtr, Xtest[i,:])))**2)
        
    mse = loss/N
      
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.flatten()
    w = w[:,np.newaxis]
    error = 0
    error_grad = 0
    #error = 0.5 * ((np.sum(y - np.dot(np.transpose(w), X)))**2) + 0.5*lambd*np.dot(np.transpose(w), w)
    #error = 0.5 * ((np.sum(y - np.dot(X, w)))**2) + 0.5*lambd*np.dot(np.transpose(w), w)
    error = 0.5 * (np.dot(np.transpose(y - np.dot(X, w)),(y - np.dot(X, w)))) + 0.5*lambd*np.dot(np.transpose(w), w)
    
    error_grad = np.dot(np.transpose(X), np.dot(X,w) - y) + (lambd*w)
    error_grad = error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 

    # IMPLEMENT THIS METHOD
    
    N = x.shape[0]
    Xd = np.zeros((N,p+1))  
    for power in range(p+1):
        #Xd[power:,] = x**power
        Xd[:,power] = x**power
           
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA'
)
plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

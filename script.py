
# coding: utf-8

# In[1]:

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


# In[2]:

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    #Dim of X: 150 by 2
    #Dim of y: 150 by 1
   
    means = np.zeros(shape=(X.shape[1],5))
    covmat = np.zeros(shape=(X.shape[1],X.shape[1]))
     
    for k in range (1,6):
            elements = X[y[:,0]==k]
            elements_mean = np.mean(elements,axis=0)
            means[:,k-1] = elements_mean
    
    
    covmat = np.cov(X,rowvar=0)
    return means,covmat


# In[3]:

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    means = np.zeros(shape=(X.shape[1],5))
    covmat = np.zeros(shape=(X.shape[1],X.shape[1]))
    #covmats = np.zeros(shape=(2,2))
    covmats = list()
     
    for k in range (1,6):
            elements = X[y[:,0]==k]
            elements_mean = np.mean(elements,axis=0)
            means[:,k-1] = elements_mean
            covmat = np.cov(elements,rowvar=0)
            covmats.append(covmat)
    #print("Mean",means)
    
    #covmat = np.cov(X,rowvar=0)
    #print("COV",covmat)
    #print("COVvvv",covmats)
    #print(len(covmats))
    return means,covmats


# In[4]:

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    b = np.zeros(shape=(Xtest.shape[0],5))
    c = np.zeros(shape=(Xtest.shape[0],1))
    acc = 0
    #print(Xtest.shape[0])
    covmat_inv = np.linalg.inv(covmat)
    for i in range(0,5):
        for j in range(0,Xtest.shape[0]):
            b[j,i] = np.dot(np.dot(Xtest[j,:]-np.transpose(means[:,i]),covmat_inv),(Xtest[j,:]-np.transpose(means[:,i])))
    
    c = (np.argmin(b,axis=1)) + 1
    
    for i in range(0,Xtest.shape[0]):
        if(ytest[i] == c[i]):
            acc = acc + 1
    ypred = c.reshape(Xtest.shape[0],1)
    #print(ypred)
    return acc,ypred


# In[5]:

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    b = np.zeros(shape=(Xtest.shape[0],5))
    c = np.zeros(shape=(Xtest.shape[0],1))
    acc = 1
    covmaat = np.array(covmats)
    #covmat_inv = np.linalg.inv(covmats[i])
    for i in range(0,5):
        for j in range(0,Xtest.shape[0]):
            p = np.dot(np.dot(Xtest[j,:]-np.transpose(means[:,i]),np.linalg.inv(covmaat[i])),(Xtest[j,:]-np.transpose(means[:,i])))
            #p = - p/2
            b[j,i] = 1/(2*pi)*np.sqrt(np.linalg.det(covmaat[i])) * np.exp(p)
    
    c = (np.argmin(b,axis=1)) + 1
    
    for i in range(0,Xtest.shape[0]):
        if(ytest[i] == c[i]):
            acc = acc + 1
    ypred = c.reshape(Xtest.shape[0],1)
    #print(ypred)
    
    return acc,ypred


# In[36]:

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD    
    inv_of_xtransx = np.linalg.inv(np.dot(np.transpose(X),X))
    xtransy = np.dot(np.transpose(X),y)
    w = np.dot(inv_of_xtransx,xtransy)
    #print("OLERidge regression wt ",w)
    return w


# In[37]:

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    #print("length of X Ridge", X.shape[1])
    w = np.dot(np.linalg.inv(np.multiply(lambd,np.identity(X.shape[1])) + np.dot(np.transpose(X),X)), (np.dot(np.transpose(X),y)))
    

    
    #print("Ridge regression wt ",w)
    return w


# In[8]:

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    xdotw = np.dot(Xtest,w)
    mse = (np.dot(np.transpose(ytest - xdotw), (ytest - xdotw))) / N
    
    #implementation for training data
    #xdotw_tr = np.dot(X,w)
    
    #mse = np.dot(np.transpose(ytest - np.dot(np.transpose(w),Xtest)),(ytest - np.dot(np.transpose(w),Xtest))) * 0.5
    #print (mse)
    return mse


# In[9]:

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    
    #Error
    
    w = np.reshape(w, (w.size,1))
    #print(len(X))
    xdotw = np.dot(X,w)
    secondTerm = (np.dot(np.transpose(w),w)) * lambd
    
    secondTerm = secondTerm/2  
    b = np.subtract(y,xdotw)
    firstTerm = np.dot(np.transpose(b),b)
  
    error = firstTerm + secondTerm
    #print(error)
    
    #Error Gradient
    xTransX = np.dot(np.transpose(X),X)
    wTransX = np.dot(np.transpose(w),xTransX)
    yTransX = np.dot(np.transpose(y),X)
    N = X.shape[0]
    a = np.divide(np.subtract(wTransX, yTransX),N)
    b = np.dot(np.transpose(w),lambd)
    error_grad = np.add(a,b)
    #print(error_grad.shape[1])
    error_grad = np.reshape(error_grad, (error_grad.size,1))
    #print("Error Grad: ",error_grad)
    error_grad = error_grad.flatten()
    #print("Error Grad: flattened    ",error_grad)
    
   
    return error, error_grad


# In[10]:

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xd = np.ones(shape = (N,p+1))
    
    #for(i in range (1,p+1)):
    #    Xd[:,i] = np.power(x[:,i],i)
    
    for i in range (0,N):
        for j in range (1, p+1):
            Xd[i][j] = np.power(x[i],j)
    
    return Xd


# In[41]:

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
#print(zldares.shape[0])
#print(zldares.shape[1])
#print(x1.shape[0])
#print(x2.shape[0])
#print(zldares)
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

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

mle_te = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_te_i = testOLERegression(w_i,Xtest_i,ytest)

#print('For test data:')
#print('MSE without intercept '+str(mle_te))
#print('MSE with intercept '+str(mle_te_i))

#For training data
Xi = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtesti = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

w_tr = learnOLERegression(X,y)

mle_tr = testOLERegression(w_tr,X,y)

w_tr_i = learnOLERegression(Xi,y)
mle_tr_i = testOLERegression(w_tr_i,Xtesti,y)

#print("2. OLE regression wt ",w)
#print('For training data:')
#print('MSE without intercept '+str(mle_tr))
#print('MSE with intercept '+str(mle_tr_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
optimisedLambda_Train = list()
optimisedLambda_Test = list()




lambdaVsMse_Train = np.zeros(shape=(1,2))
#print("3 Lamda:")
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    optimisedLambda_Train.append([lambd,mses3_train[i]])
    optimisedLambda_Test.append([lambd,mses3[i]])
    i = i + 1
    #print(lambd)
#print("3 Ridge regression wt ",w_l)
#print("3) MSE Train: \n", mses3_train)
#print("3)MSE Test: \n", mses3)

#print("3) Optimised Lambda Test Values ",optimisedLambda_Test)
#print("3) Optimised Lambda Train Values ", optimisedLambda_Train)

#print("3) Optimised Lambda Test  ", min(map(lambda x: x[1], optimisedLambda_Test )))
#print("3) Optimised Lambda Train  ", min(map(lambda x: x[1], optimisedLambda_Train )))
#print(min(map(lambda x: x[0], optimisedLambda_Test )))
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
opts = {'maxiter' : 700}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
optimisedLambda_Train4 = list()
optimisedLambda_Test4 = list()
for lambd in lambdas:
    #print("i   ", i)
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    optimisedLambda_Train4.append([lambd,mses4_train[i]])
    optimisedLambda_Test4.append([lambd,mses4[i]])
    i = i + 1

#print("4) Optimised Lambda Test  ",min(map(lambda x: x[1], optimisedLambda_Test4 )))
#print("4) Optimised Lambda Train  ",min(map(lambda x: x[1], optimisedLambda_Train4 )))    
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
lambda_opt = lambdas[np.argmin(mses3)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
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

#print("5) Optimised Lambda Test  ",min(map(lambda x: x[1], mses5 )))
#print("5) Optimised Lambda Train  ",min(map(lambda x: x[1], mses5_train )))
#print("5) MSE Train: \n", mses3_train)
#print("5)MSE Test: \n", mses3)
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


# In[ ]:




# In[ ]:




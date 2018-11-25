import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE


w=train(Xtrain, Ytrain, alpha=0.15, n_epoch=100)
print('Alpha=0.15')
print('Epoch=100')
y_pred_train=Xtrain.dot(w)
train_loss=compute_L(y_pred_train, Ytrain)
print('Train Loss')
print(train_loss)
y_pred_test=Xtest.dot(w)
test_loss=compute_L(y_pred_test, Ytest)
print('Test Loss')
print(test_loss)


'''
alphas=[0.35, 0.30, 0.25, 0.2, 0.15,0.10,0.05]
loss_trains=[]
loss_tests=[]
for alpha_test in alphas:
    w=train(Xtrain, Ytrain, alpha=alpha_test, n_epoch=100)
    y_pred_train=Xtrain.dot(w)
    train_loss=compute_L(y_pred_train, Ytrain)
    print('Train Loss')
    print(train_loss)
    loss_trains.append(train_loss[0,0])
    y_pred_test=Xtest.dot(w)
    test_loss=compute_L(y_pred_test, Ytest)
    print('Test Loss')
    print(test_loss)
    loss_tests.append(test_loss[0,0])
print(loss_tests)
print(loss_trains)

import matplotlib.pyplot as plt
#plt.ylim(-.005,.005)
plt.scatter(alphas,loss_tests)
plt.show()
'''

'''    
epochs=[30, 50, 70 ,90 , 120]
loss_trains=[]
loss_tests=[]
for epoch in epochs:
    w=train(Xtrain, Ytrain, alpha=0.1, n_epoch=epoch)
    y_pred_train=Xtrain.dot(w)
    train_loss=compute_L(y_pred_train, Ytrain)
    print('Train Loss')
    print(train_loss)
    loss_trains.append(train_loss[0,0])
    y_pred_test=Xtest.dot(w)
    test_loss=compute_L(y_pred_test, Ytest)
    print('Test Loss')
    print(test_loss)
    loss_tests.append(test_loss[0,0])
print(loss_tests)
print(loss_trains)

import matplotlib.pyplot as plt
#plt.ylim(-.005,.005)
plt.scatter(epochs,loss_tests)
plt.show()
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

alphas=[0.25, 0.2,0.175, 0.15, 0.125,0.10,0.075, 0.05,0.025, 0.015]
epochs=[ 50, 70 ,90 , 110,130, 150 ]
loss_trains=[]
loss_tests=[]
alpha_comb=[]
epoch_comb=[]
for epoch in epochs:
    for alpha in alphas:
        w=train(Xtrain, Ytrain, alpha=alpha, n_epoch=epoch)
        y_pred_train=Xtrain.dot(w)
        train_loss=compute_L(y_pred_train, Ytrain)
        loss_trains.append(train_loss[0,0])
        y_pred_test=Xtest.dot(w)
        test_loss=compute_L(y_pred_test, Ytest)
        loss_tests.append(test_loss[0,0])
        alpha_comb.append(alpha)
        epoch_comb.append(epoch)
        

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=15)


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
ax.scatter( alpha_comb, epoch_comb,loss_tests)

ax.set_xlabel('Alpha')
ax.set_ylabel('Epoch')
ax.set_zlabel('Loss')


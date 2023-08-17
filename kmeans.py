import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Iris.csv",header = 'infer').values

x = data[:,1:-1]
y = data[:,-1]

test_split=float(input('enter the number between 0 and 1 to specify how much data is required:'))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_split)
k=int(input("Enter number of clusters you want to divide your data in i.e k:"))
n=int(input('Enterno of iteration you want to learn algorithm:'))
centroids=np.zeros(shape=(k,x_train.shape[1]))
print(centroids)
per = np.random.permutation(x_train.shape[0])
for i in range(k):
     centroids[i,:] = x_train[per[i],:]
for it in range(n):
    dist = np.zeros(shape = (k, x_train.shape[0]))
for i in range(k):
    dist[i,:] = np.sqrt(np.sum((x_train-centroids[i,:])**2, axis = 1))
membership = np.argmin(dist , axis = 0)
for i in range(k):
    centroids[i,:] = np.mean(x_train[membership == i,:], axis = 0)
print("Centroids after " + str(n) + " iteration")
print("Centroids")
dist = np.zeros(shape = (k, x_test.shape[0]))
for i in range(k):
    dist[i] = np.sqrt(np.sum((x_test-centroids[i])**2, axis = 1))
membership = np.argmin(dist,axis = 0)

print(y_test.astype(int))
print(membership)

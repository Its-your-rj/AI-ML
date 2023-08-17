import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Iris.csv",header = 'infer').values

X = data[:,1:-1]
Y = data[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y)

nclasses = np.unique(Y_train).shape[0]

print(nclasses)

dist = np.zeros(shape = X_train.shape[0])
print(len(dist))
pred = np.zeros(shape = X_test.shape[0])
print(len(dist))
classvotes = np.zeros(shape = nclasses)
print(classvotes)

k = int(input("Enter the number of nearest neighbours to be used, i.e, k : "))

for i in range(X_test.shape[0]):
    dist = np.sqrt(np.sum((X_train-X_test[i])**2,axis=1))
    
    kminind = np.argpartition(dist,k)[0:k]
    invdist = 1/(dist+10e-20)
    denom = sum(invdist[kminind])
    for j in range(k):
        classvotes[int(Y_train[kminind[j]])] += invdist[kminind[j]]
        print(classvotes)
    classvotes/= denom
    
    pred[i] = np.argmax(classvotes)
    print(pred[i])
    
def calc_acc(Y_pred, Y_true) : 
    return np.sum((Y_pred).astype(int) == (Y_true).astype(int))/Y_pred.shape[0]

accuracy = calc_acc(pred,Y_test)

print("Accuracy", accuracy)

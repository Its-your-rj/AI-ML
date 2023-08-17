import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("BostonHousing.csv",header = 'infer').values

x = data[:,1:-1]
y = data[:,-1]

test_split=float(input('enter the number between 0 and 1 to specify how much data is required:'))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_split)
dist=np.zeros(shape=x_train.shape[0])
pred=np.zeros(shape=x_test.shape[0])

k=int(input("enter no of nearest neighbour to be used:"))
for i in range(x_test.shape[0]):
    dist=np.sqrt(np.sum((x_train-x_test[i])**2,axis=1))
    kminind=np.argpartition(dist,k)[0:k]
    invdist=1/(dist+10e-20)
    denom=sum(invdist[kminind])
    pred[i]=np.dot(invdist[kminind]/denom,y_train[kminind])
    
    
def MAE(pred,y_test):
    return np.mean(abs(pred-y_test))
    
def MSE(pred,y_test):
     return np.mean((pred-y_test)**2) 
    
def MAPE(pred,y_test):
    return np.mean(abs((pred-y_test)/y_test))
    
    
    
mae=MAE(pred,y_test)
mse=MSE(pred,y_test)
rmse=np.sqrt(mse)
mape=MAPE(pred,y_test)


print('MAE:',mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("MAPE:",mape)

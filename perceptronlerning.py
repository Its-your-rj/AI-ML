import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ds1=np.random.randint(low=1,high=10,size=(50,2))
print(ds1)
ds2=-ds1
x=np.concatenate((ds1,ds2),axis=0)
print(x)

y=np.ones(shape=100)
print(y)
y[:50]=0
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y)

plt.scatter(x=x_train[:,0],y=x_train[:,1],c=y_train)
plt.show()
n_samples=x_train.shape[0]
n_features=x_train.shape[1]
w=np.random.uniform(0,1,size=n_features)
b=np.random.uniform(0,1,1)
n_epoch=int(input("enter name of epochs:"))
lr=0.01
for e in range(n_epoch):
    for s in range(n_samples):
        net=np.dot(x_train[s,:],w)+b
        if net>=0:
            a=1
        else:
            a=0
        error=y_train[s]-a
        w=w+lr*error*x_train[s,:]
        b=b+lr*error
        
        
net=np.dot(x_test,w)+b
pred=list(map(int,(net>=0)))

print(pred)

print("classification report:")
print(classification_report(y_true=y_test,y_pred=pred))

#clculating slope and the intercept of the decision boundary

m=-w[0]/w[1]
c=-b/w[1]


def plot_decision_boundary(x):
    for x in np.linspace(np.min(x[:,0]),np.max(x[:,0])):
        y=m*x+c
        plt.plot(x,y,linestyle="-",color='k',marker='.')
    plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
    plt.show()
plot_decision_boundary(x_train)

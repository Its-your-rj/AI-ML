import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier



data = pd.read_csv("Iris.csv",header = 'infer').values

X = data[:,1:-1]
Y = data[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y)

nclasses = np.unique(Y_train).shape[0]

print(nclasses)

dist = np.zeros(shape = X_train.shape[0])
dist = np.zeros(shape = X_test.shape[0])

classvotes = np.zeros(shape = nclasses)

k = int(input("Enter the number of nearest neighbours to be used, i.e, k : "))

model=KNeighborsClassifier(n_neighbors=k,weights='distance')
model.fit(X_train,Y_train)
pred=model.predict(X_test)

accuracy=accuracy_score(Y_test,pred)
print("accuracy:",accuracy)

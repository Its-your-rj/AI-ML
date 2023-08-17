import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', max_iter=1000)
mlp.fit(X, y)
predicted_outputs = mlp.predict(X)
print(predicted_outputs)

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('XOR Gate')
plt.show()

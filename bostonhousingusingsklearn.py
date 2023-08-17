from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

model=KNeighborsRegressor(n_neighbors=k,weights='distance')

model.fit(x_train,y_train)

pred=model.predict(x_test)

mae1=mean_absolute_error(y_test,pred)
mse1=mean_squared_error(y_test,pred)
mase1=mean_absolute_percentage_error(y_test,pred)



print("using sklearn:")
print("MAE:",mae1)
print("MSE",mse1)
print("MSE",mase1)

# Boston DataSet From SKLEARN
# MUHMMAD HANAN ASGHAR 2ND MODEL


from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

boston = datasets.load_boston()

# df = pd.DataFrame(boston.data,columns=boston.feature_names)

boston_X = boston.data

boston_X_train = boston_X[:-30]
boston_X_test = boston_X[-30:]

boston_Y_train = boston.target[:-30]
boston_Y_test = boston.target[-30:]

model = linear_model.LinearRegression()
model.fit(boston_X_train,boston_Y_train)
boston_predicted = model.predict(boston_X_test)


print("Weight : ",model.coef_)
print("Intercept : ",model.intercept_)
print("Mean Squared Error : ",mean_squared_error(boston_Y_test,boston_predicted))
print(boston_predicted)
print("Main Values")
print(boston_Y_test)



# sns.distplot(boston.target,bins=30)
# plt.show()

# print(boston['DESCR'])

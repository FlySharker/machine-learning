import pandas as pd
import matplotlib.pyplot as plt
from sklearn .datasets import fetch_california_housing
from sklearn.linear_model import LassoLars
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

housing = fetch_california_housing()
housing_df=pd.DataFrame(data=housing.data,columns=housing.feature_names)
housing_df['target']=housing.target
print(housing_df.head())
X=housing.data
y=housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=420)
reg=LassoLars().fit(X_train,y_train)
y_pred=reg.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print('均方误差',mse)
plt.scatter(y_test,y_pred,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--k')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('线性回归')
plt.show()

plt.plot(range(len(y_test)),sorted(y_test),c='black',label='Data')
plt.plot(range(len(y_pred)),sorted(y_pred),c='red',label='Predict')
plt.legend()
plt.show()
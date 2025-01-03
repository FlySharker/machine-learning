from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Lasso
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
iris=load_iris()
X=iris.data
y=iris.target
iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['target']=iris.target
print(iris_df.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=420)
model=LR().fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print('均方误差',mse)
print('准确率',acc)
plt.scatter(y_test,y_pred,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--k')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('逻辑回归回归')
plt.show()

plt.plot(range(len(y_test)),sorted(y_test),c='black',label='Data')
plt.plot(range(len(y_pred)),sorted(y_pred),c='red',label='Predict')
plt.legend()
plt.show()


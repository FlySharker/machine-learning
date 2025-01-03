import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot  # 矩阵点乘
from sklearn .datasets import fetch_california_housing
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
tmp=0
mse=0
cnt=10321
housing = fetch_california_housing()
housing_df=pd.DataFrame(data=housing.data,columns=housing.feature_names)
housing_df['bias']=1
housing_df['target']=housing.target
print(housing_df.head())
X = housing_df.iloc[0:cnt, 0: 9]
Y = housing_df.iloc[0:cnt, -1]
a = dot(dot(inv(np.dot(X.T, X)), X.T), Y)# 最小二乘法求解公式
list_pred=[]
list_act=[]
for i in range(cnt-2):
    y_pred=a[0]*housing_df.iloc[cnt+i,0]+a[1]*housing_df.iloc[cnt+i,1]+a[2]*housing_df.iloc[cnt+i,2]+a[3]*housing_df.iloc[cnt+i,3]+a[4]*housing_df.iloc[cnt+i,4]+a[5]*housing_df.iloc[cnt+i,5]+a[6]*housing_df.iloc[cnt+i,6]+a[7]*housing_df.iloc[cnt+i,7]+a[8]*housing_df.iloc[cnt+i,8]
    y_act=housing_df.iloc[cnt+i,-1]
    tmp=tmp+(y_pred-y_act)**2
    list_pred.append(y_pred)
    list_act.append(y_act)
mse=tmp/(cnt-1)
print(mse)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('最小二乘法')
plt.plot(range(len(list_act)),sorted(list_act),c='black',label='Data')
plt.plot(range(len(list_pred)),sorted(list_pred),c='red',label='Predict')
plt.legend()
plt.show()
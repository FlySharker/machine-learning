from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

iris=load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

model=MultinomialNB().fit(X_train,y_train)
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)

print("测试集准确率：", acc)

plt.scatter(y_test,y_pred,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--k')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('贝叶斯')
plt.show()

plt.plot(range(len(y_test)),sorted(y_test),c='black',label='Data')
plt.plot(range(len(y_pred)),sorted(y_pred),c='red',label='Predict')
plt.legend()
plt.show()




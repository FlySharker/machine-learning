#导入所需的模块
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression

#导入Keras提供的数据集MNIST模块
(x_train_image,y_train_label), (x_test_image,y_test_label) = mnist.load_data()

#转化（reshape）为一维向量，其长度为784，并设为Float数。
x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

#将数据归一化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

#传递训练模型的参数
print(time.strftime('%Y-%m-%d %H:%M:%S'))
clf = LogisticRegression(max_iter=10000)

# 进行模型训练
t1 = time.time()
clf.fit(x_Train_normalize, y_train_label)
t2 = time.time()
Logicfit = float(t2-t1)
print("Time taken: {} seconds".format(Logicfit))

predictions = [int(a) for a in clf.predict(x_Test_normalize)]
#混淆矩阵
print(confusion_matrix(y_test_label, predictions))
#f1-score,precision,recall
print(classification_report(y_test_label, np.array(predictions)))

#计算准确度
print('accuracy=', accuracy_score(y_test_label, predictions))
print(time.strftime('%Y-%m-%d %H:%M:%S'))
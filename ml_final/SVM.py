import os
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

def read_images_from_folder_train(folder_path,list):
    images = []
    path_list=os.listdir(folder_path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))  # 对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
    for filename in path_list:
        tmp=int(filename.rsplit('.', 1)[0])
        if filename.endswith(".png") and tmp in list:  # 根据需要调整文件扩展名
            image = Image.open(os.path.join(folder_path, filename))
            image = image.convert("L")  # 转换为灰度图像
            image = np.array(image)  # 将图像转换为NumPy数组
            images.append(image)
    return images


def read_images_from_folder_test(folder_path):
    images = []
    path_list = os.listdir(folder_path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))  # 对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
    for filename in path_list:
        if filename.endswith(".png"):  # 根据需要调整文件扩展名
            image = Image.open(os.path.join(folder_path, filename))
            image = image.convert("L")  # 转换为灰度图像
            image = np.array(image)  # 将图像转换为NumPy数组
            images.append(image)
    return images

def read_labels_from_folder(folder_path):
    labels=[]
    with open(folder_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.split()
            labels.append(int(item[1]))

    return labels

def read_idx_from_folder(folder_path):
    idxs=[]
    with open(folder_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.split()
            idxs.append(int(item[0]))

    return idxs

print(time.strftime('%Y-%m-%d %H:%M:%S'))
train_folder_path = "D:\\p_y\\dataset\\MNIST\\mnist_dataset\\train"
test_folder_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test'
train_labels_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt'
test_labels_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test_labs.txt'
y_train_idx=read_idx_from_folder(train_labels_path)
train_images = read_images_from_folder_train(train_folder_path,y_train_idx)
test_images = read_images_from_folder_test(test_folder_path)
y_train_label=read_labels_from_folder(train_labels_path)
y_test_label=read_labels_from_folder(test_labels_path)

print(time.strftime('%Y-%m-%d %H:%M:%S'))
#转化（reshape）为一维向量，其长度为784，并设为Float数。
train_images=np.array(train_images)
test_images=np.array(test_images)
y_train_label=np.array(y_train_label)
y_test_label=np.array(y_test_label)
x_Train =train_images.reshape(2000, 784).astype('float32')
x_Test =test_images.reshape(10000, 784).astype('float32')
print(time.strftime('%Y-%m-%d %H:%M:%S'))
#将数据归一化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
print(time.strftime('%Y-%m-%d %H:%M:%S'))
clf = svm.SVC()
clf.fit(x_Train_normalize, y_train_label)
print(time.strftime('%Y-%m-%d %H:%M:%S'))
predictions = [int(a) for a in clf.predict(x_Test_normalize)]
#混淆矩阵
print(confusion_matrix(y_test_label, predictions))
#f1-score,precision,recall
print(classification_report(y_test_label, np.array(predictions)))
#计算准确度
print('accuracy=', accuracy_score(y_test_label, predictions))
print(time.strftime('%Y-%m-%d %H:%M:%S'))

import numpy as np
from sklearn import svm
from PIL import Image
import time

train_folder_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\train'
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt', 'r') as f:
    lines = f.readlines()
data1 = [int(x) for line in lines for x in line.split()[0:1]]
data2 = [int(x) for line in lines for x in line.split()[1:2]]
images_train=[]
for data in data1:
    image = Image.open(train_folder_path + '\\' + str(data) + '.png')
    image_array = np.array(image)
    image_array = image_array.ravel()
    images_train.append(image_array)
model=svm.SVC()
model.fit(images_train, data2)
test_folder_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test'
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test_labs.txt', 'r') as f:
    lines = f.readlines()
data3 = [int(x) for line in lines for x in line.split()[0:1]]
data4 = [int(x) for line in lines for x in line.split()[1:2]]
images_test=[]
for data in data3:
    image = Image.open(test_folder_path + '\\' + str(data) + '.png')
    image_array = np.array(image)
    image_array = image_array.ravel()
    images_test.append(image_array)
z = model.predict(images_test)
classes= [0] * 20
acc=[0]*10
for a,b in zip(data4,z):
    if a == b:
        classes[a]+=1
    else:
        classes[a+10]+=1
for i in range(10):
    acc[i]=classes[i]/(classes[i]+classes[i+10])
    print('class',i,":",acc[i])
print('准确率:', sum(acc) / len(acc))
print(time.strftime('%Y-%m-%d %H:%M:%S'))

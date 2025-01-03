# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
from mnist_pytorch import CNN
from cleanlab.classification import CleanLearning
import os

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = Image.open(os.path.join(folder_path, filename))
            image = image.convert("L")
            image = np.array(image)
            images.append(image)
    return images

def read_labels_from_folder(folder_path,x):
    labels=[]
    with open(folder_path,'r') as f:
        lines=f.readlines()
        for line in lines:
            item=line.split()
            labels.append(int(item[x]))
    return labels

if __name__ == '__main__':
    cnn = CNN(epochs=5, log_interval=100, loader='train')
    x_train = []
    train_labels_path = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt'
    x_train=read_labels_from_folder(train_labels_path,0)
    y_train_label = read_labels_from_folder(train_labels_path,1)
    x_train=np.array(x_train)
    y_train_label = np.array(y_train_label)
    label_issues_info = CleanLearning(clf=cnn).find_label_issues(x_train, y_train_label)
    print(label_issues_info)
    label_issues_info.to_csv('D:\\p_y\\ml\\ml_final\\x.csv',index=False )



# -*- coding:utf-8 -*-
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch import optim

def Myloader(path):
    return Image.open(path).convert('L')

# get a list of paths and labels.
def init_process(path1,path2):
    i=0
    data = []
    with open(path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            a = line.split()
            x=a[1]
            i=int(a[0])
            data.append([path2 % i, x])
    return data

class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # normalization
    ])
    path1 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt'
    path2 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\train\\%d.png'
    data1 = init_process(path1, path2)
    path3='D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test_labs.txt'
    path4 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test\\%d.png'
    data2 = init_process(path3,path4)
    # np.random.shuffle(data1)
    # np.random.shuffle(data2)
    train_data, test_data = data1, data2
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=200, shuffle=True, num_workers=0)
    return Dtr, Dte

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=1,
                out_channels=16,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=16,
                out_channels=32,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x=self.conv(x)
        x = self.conv2(x)
        x=x.view(x.size(0),-1)
        out=self.out(x)
        return out

mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

def train(epoch):
    train_dataloader, test_dataloader = load_data()
    print('图片数量：', len(train_dataloader.dataset))
    print('batch数量：', len(train_dataloader))
    train_loss_list=[]
    for idx, (data, target) in enumerate(train_dataloader):
        target=np.array(target)
        target=target.astype(float)
        target=torch.from_numpy(target).long()
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())
    return train_loss_list


def test():
    test_loss = 0
    correct = 0
    classes=('0','1','2','3','4','5','6','7','8','9')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    train_dataloader, test_dataloader = load_data()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = np.array(target)
            target = target.astype(float)
            target = torch.from_numpy(target).long()
            output = mnist_net(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            for label, prediction in zip(target, pred):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader)
    print('\nTest set: Avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))

if __name__ == '__main__':
    train(1)
    test()

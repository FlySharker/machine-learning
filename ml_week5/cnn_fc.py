import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from torch import optim
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 超参数
EPOCH = 1  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
train_batch_size = 128
test_batch_size = 128

def get_dataloader(train):
   # assert isinstance(train, bool), "train 必须是bool类型"

    # 准备数据集，其中0.1307，0.3081为MNIST数据的均值和标准差，这样操作能够对其进行标准化
    # 因为MNIST只有一个通道（黑白图片）,所以元组中只有一个值
    dataset = MNIST('./data', train=train , download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                         ]))
    # 准备数据迭代器
    batch_size = train_batch_size if train else test_batch_size
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Sequential(
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
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        out=self.out(x)
        return out


mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

def train(epoch):
    mode = True
    mnist_net.train(mode=mode)
    train_dataloader = get_dataloader(train=mode)
    print('图片数量：', len(train_dataloader.dataset))
    print('batch数量：', len(train_dataloader))
    train_loss_list=[]
    for idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = loss_func(output, target)  # 对数似然损失
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())
    return train_loss_list

def draw_loss(loss):
    x=range(0,47)
    y=loss
    plt.plot(x,y,'-')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train_Loss')
    plt.show()

def test():
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_dataloader = get_dataloader(train=False)
    print('图片数量：', len(test_dataloader.dataset))
    print('batch数量：', len(test_dataloader))
    with torch.no_grad():
        for data, target in test_dataloader:
            output = mnist_net(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    for i in range(1):  # 模型训练5轮
        loss_list = train(i)
        draw_loss(loss_list)
        test()

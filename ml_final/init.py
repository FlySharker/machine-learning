import pandas as pd

data=[]
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\train_labs.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        x=line.split()
        a=x[1]
        data.append(a)

f.close()

with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        x=line.split()
        a=x[0]
        b=x[1]
        data[int(a)]=b

f.close()

with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\final_train_labs.txt','w',encoding='utf-8') as f:
    i=0
    for i in range(55000):
        f.write(str(i))
        f.write(' ')
        f.write(str(data[i]))
        f.write('\n')




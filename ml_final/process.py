import pandas as pd

data_err=[]
data_idx=[]
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\50%_incorrect_train_labs.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        x=line.split()
        data_err.append(int(x[1]))
        data_idx.append(int(x[0]))
f.close()

data_pre=[]
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\less_train_labs.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        x=line.split()
        data_pre.append(int(x[1]))
f.close()

is_bool=[]
sum=0
for i in range(2000):
    if data_pre[i]!=data_err[i]:
        sum=sum+1
        is_bool.append('True')
    else:
        is_bool.append('False')
print(sum)
print(is_bool)

df=pd.read_csv('D:\\p_y\\ml\\ml_final\\x.csv')
issue=df.iloc[:,0]
issue=issue.tolist()
for i in range(2000):
    issue[i]=str(issue[i])

idx_err=[]
cnt=0
for i in range(2000):
    if is_bool[i]=='True' and issue[i]=='True':
        idx_err.append(i)
        cnt=cnt+1

print(cnt)

m=0
with open('D:\\p_y\\dataset\\MNIST\\mnist_dataset\\final_train_labs_less_noise.txt','w',encoding='utf-8') as f:
    for m in range(2000):
        if m not in idx_err:
            f.write(str(data_idx[m]))
            f.write(' ')
            f.write(str(data_pre[m]))
            f.write('\n')
        m = m + 1




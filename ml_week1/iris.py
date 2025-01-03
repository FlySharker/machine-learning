from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
df['target_name']='x'
for i in range(0,150):
    if df.iloc[i,4]==0:
        df.iloc[i,5]='setosa'
    elif df.iloc[i,4]==1:
        df.iloc[i, 5] = 'versicolor'
    elif df.iloc[i,4]==2:
        df.iloc[i,5]='virginica'
print(df)

sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target_name', style='target_name', data=df)
plt.title('Iris dataset - Sepal length vs Sepal width')
plt.show()




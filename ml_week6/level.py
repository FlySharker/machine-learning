from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score, fowlkes_mallows_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

iris=load_iris()
X=iris.data
y=iris.target

model=AgglomerativeClustering(linkage='ward', n_clusters=3).fit(X)
y_pred = model.labels_
sil=silhouette_score(X,y_pred)
ch=calinski_harabasz_score(X,y_pred)
fow=fowlkes_mallows_score(y,y_pred)
print('预测标签：{}\n'.format(y_pred))
print('畸变程度：{}\n聚类分数：{}\nFMI：{}\n'.format(sil,ch,fow))

tsne = TSNE(n_components=2,init='random',random_state=177).fit(X)
df = pd.DataFrame(tsne.embedding_)
df['labels'] = y_pred
df.columns=['x','y','labels']
print(df)

plt.figure(dpi=200, figsize=(6,4))
sns.scatterplot(x='x', y='y', hue='labels', style='labels', data=df)
plt.title('聚类效果')
plt.show()

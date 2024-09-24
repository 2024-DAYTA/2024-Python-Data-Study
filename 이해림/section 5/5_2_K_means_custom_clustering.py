# 데이터 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/zzhining/ml_basic/main/dataset/Mall_Customers.csv')
df.head()

df.shape

# 데이터 탐색
df.describe()

df['Gender'].value_counts() #gender 범주형

sns.countplot(x=df['Gender'])

sns.barplot(x= 'Gender',  y = 'Annual Income (k$)', data = df)

sns.boxplot(x= 'Gender',  
            y = 'Spending Score (1-100)', 
            data = df)

sns.scatterplot(x= 'Age',  
                y = 'Annual Income (k$)', 
                data = df)

sns.scatterplot(x= 'Age',  
                y = 'Spending Score (1-100)',
                data = df)

sns.scatterplot(x= 'Annual Income (k$)', 
                y = 'Spending Score (1-100)', 
                data = df)

# 데이터 전처리

# CustomerID 컬럼 삭제
df.drop(['CustomerID'], axis = 1,inplace = True)

# Gender 컬럼 수치형 변수로 변경
df['Gender'] =df['Gender'].apply(lambda x : 0 if x =='Male' else 1)

df.head()

# 스케일 변환
from sklearn.preprocessing import StandardScaler

X= df.iloc[:,1:]
sc = StandardScaler()
X = sc.fit_transform(X)

#K-means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

model = KMeans(n_clusters= 2, random_state=42)
y_preds = model.fit_predict(X)
df['cluster'] = y_preds

print('실루엣 점수: {0:.4f}'.format(silhouette_score(X, y_preds))) 

sns.scatterplot(x= 'Age',  y = 'Annual Income (k$)', 
                data = df, hue = 'cluster')

# 최적의 군집 수 찾기
silhouette_avg = []
for k in range(10):
    model = KMeans(n_clusters= k+2, random_state=42)
    y_preds = model.fit_predict(X)
    score = silhouette_score(X, y_preds)
    silhouette_avg.append(score)
    print("군집개수: {0}개, 평균 실루엣 점수: {1:.4f}"
    .format(k+2, score))

plt.plot(range(2,12), silhouette_avg, 'bo--')
plt.xlabel('# of clusters')
plt.ylabel('silhouette_avg')
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score

def plotSilhouette(n_clusters, y_preds):   
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    y_lower = 10
    silhouette_avg = silhouette_score(X, y_preds)   
    sample_silhouette_values = silhouette_samples(X, y_preds)
    print("군집개수: {0}개, 평균 실루엣 점수: {1:.4f}".format(k, silhouette_avg))

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[y_preds == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  

    ax1.set_title("The silhouette plot for the {0} clusters.".format(n_clusters))
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

import matplotlib.cm as cm
k_list = [4, 5, 6]

for k in k_list:
    model = KMeans(n_clusters= k, random_state=42)
    y_preds = model.fit_predict(X)
    plotSilhouette(k, y_preds)

model = KMeans(n_clusters= 6, random_state=42)
y_preds = model.fit_predict(X)
df['cluster'] = y_preds

sns.scatterplot(x = 'Annual Income (k$)',
                y = 'Spending Score (1-100)', 
                data = df, hue = 'cluster', 
                palette="deep")

sns.swarmplot(x = 'cluster', y='Age', data = df)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(x, y, z, category, title) :
    axes3d = Axes3D(plt.figure(figsize=(8, 6))) 
    axes3d.scatter(xs = x, ys = y, zs = z, c = category)
    axes3d.set_title(title)

plot3D(df[df.columns[1]], df[df.columns[2]], 
       df[df.columns[3]], df['cluster'], 'k=6')
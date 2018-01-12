import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans
import csv
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import sys

def tsne_plot(data, predict_labels):

    #tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, verbose=1)
    #new_values = tsne_model.fit_transform(data)
    
    pca2 = decomposition.PCA(n_components=2, whiten=True,svd_solver="full")
    pca2.fit(data)


    x = []
    y = []

    #for value in new_values:
    for value in pca2.transform(data):
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        if i<5000:
            plt.scatter(x[i],y[i], facecolors='none', edgecolors='r')
        else:
            plt.scatter(x[i],y[i], facecolors='none', edgecolors='b')
    plt.savefig('imagevec_real.png',dpi=500)
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        if predict_labels[i]==0:
            plt.scatter(x[i],y[i], facecolors='none', edgecolors='r')
        else:
            plt.scatter(x[i],y[i], facecolors='none', edgecolors='b')
    plt.savefig('imagevec.png',dpi=500)

X = np.load(sys.argv[1])
X = X.astype(float)/255
print(X.shape)
pca = decomposition.PCA(n_components=400, whiten=True,svd_solver="full")
pca.fit(X.T)
X_r = (pca.components_).T
#X_r = pca.transform(X)
print(X_r.shape)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_r)

ans = []
labels = kmeans.labels_
'''
data = []
predict_labels = []
Xv = np.load('data/visualization.npy')
Xv = Xv.astype(float)/255

for i in range(Xv.shape[0]):
    feature = pca.transform(Xv[i].reshape(1,-1))
    label = kmeans.predict(feature)
    print("label : ", label)
    predict_labels.append(label)
    data.append(feature[0])  
data = np.array(data, dtype=np.float64)
print(data.shape)

tsne_plot(data, predict_labels)
'''
test_data_path = sys.argv[2]
text = open(test_data_path, 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for i,r in enumerate(row):
    if i == 0:
        continue
    a = labels[int(r[1])]
    b = labels[int(r[2])]
    ans.append(int(a==b))

output_path = sys.argv[3]
print('=====Write output to %s =====' % output_path)
with open(output_path, 'w') as f:
    f.write('ID,Ans\n')
    for i, v in  enumerate(ans):
        f.write('%d,%d\n' %(i, v))


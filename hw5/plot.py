import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Add, Dot, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
#from tsne import bh_sne
from sklearn.manifold import TSNE

m2t = dict()

type_dict = dict()
text = open('data/movies.csv', 'r', encoding='big5', errors='ignore')
row = csv.reader( (line.replace('::','^') for line in text) , delimiter="^")
all_type = 0
for i,r in enumerate(row):
    if i == 0:
        continue
    a = r[2].split('|')
    if len(a) > 1:
        continue
    m2t[int(r[0])-1] = r[2]
    if a[0] not in type_dict:
        type_dict[a[0]] = all_type
        all_type += 1
print(type_dict)

def load_data(test_data_path):
    users = []
    items = []
    rates = []
    x = []
    y = []

    text = open(test_data_path, 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for i,r in enumerate(row):
        if i == 0:
            continue
        if i > 10000:
            break
        users.append(int(r[1])-1)
        items.append(int(r[2])-1)
        #rates.append(int(r[3]))
        
        if items[-1] in m2t :
            print(m2t[items[-1]])
            x.append(items[-1])
            y.append(type_dict[ m2t[items[-1]] ])
    return np.array(users, dtype=int), np.array(items, dtype=int), np.array(rates, dtype=float),x ,y

model = load_model(sys.argv[1])
model.summary()
data_path = 'data/train.csv'
users, items, rates, x, y = load_data(data_path)
output_path = sys.argv[2]

result = model.predict({'user_input':users, 'item_input':items}, batch_size = 1024)
result = result*1.11689766115 + 3.58171208604
y_ = result

movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print(movie_emb.shape)
def draw(x,y):
    y = np.array(y)
    for i in range(len(x)):
        x[i] = movie_emb[x[i]]
    x = np.array(x,dtype=np.float64)
    print(y)
    #vis_data = bh_sne(x)
    vis_data = TSNE(n_components=2).fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('jet',len(type_dict))
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm, s=0.3)
    plt.colorbar(sc)
    plt.savefig('tsne.png')
    #plt.show()
draw(x,y)
exit()
print('=====Write output to %s =====' % output_path)
with open(output_path, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(y_):
        f.write('%d,%f\n' %(i+1, v))

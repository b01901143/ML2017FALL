import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Add, Dot, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def load_data(test_data_path):
    users = []
    items = []
    rates = []

    text = open(test_data_path, 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for i,r in enumerate(row):
        if i == 0:
            continue
        users.append(int(r[1])-1)
        items.append(int(r[2])-1)
        #rates.append(int(r[3]))
        
    return np.array(users, dtype=int), np.array(items, dtype=int), np.array(rates, dtype=float)

model = load_model(sys.argv[3])
model.summary()
data_path = sys.argv[1]
users, items, rates = load_data(data_path)
output_path = sys.argv[2]

result = model.predict({'user_input':users, 'item_input':items}, batch_size = 1024)
result = result*1.11689766115 + 3.58171208604
y_ = result
print('=====Write output to %s =====' % output_path)
with open(output_path, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(y_):
        f.write('%d,%f\n' %(i+1, v))

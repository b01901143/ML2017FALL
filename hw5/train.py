# -*- coding: utf-8 -*-
import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Add, Dot, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, Adadelta
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

opt = Adam(lr=0.01, decay= 0.06)

m2t = dict()
type_list = ['Comedy', 'Action', 'Drama']
text = open('data/movies.csv', 'r', encoding='big5', errors='ignore')
row = csv.reader( (line.replace('::','^') for line in text) , delimiter="^")
for i,r in enumerate(row):
    if i == 0:
        continue
    m2t[int(r[0])-1] = r[2]

def load_data(train_data_path):
    users = []
    items = []
    rates = []

    text = open(train_data_path, 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for i,r in enumerate(row):
        if i == 0:
            continue
        users.append(int(r[1])-1)
        items.append(int(r[2])-1)
        rates.append(float(r[3]))
    users = np.array(users, dtype=int)
    items = np.array(items, dtype=int)
    rates = np.array(rates, dtype=float)
    
    randomize = np.arange(rates.shape[0])
    np.random.shuffle(randomize)
    
    users = users[randomize]
    items = items[randomize]
    rates = rates[randomize]
    
    test_len = users.shape[0] // 10
    mu = sum(rates) / rates.shape[0]
    sigma = np.std(rates)
    rates = (rates -mu)/sigma
    print(mu, sigma)
    return users[:], items[:], rates[:], np.max(users)+1, np.max(items)+1, users[:test_len], items[:test_len], rates[:test_len]

def mf_model(n_users, n_items, l_dim):
    user_input = Input(shape = [1], name='user_input')
    item_input = Input(shape = [1], name='item_input')
    user_vec = Embedding(n_users, l_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    #user_vec = Dropout(0.35)(user_vec)
    item_vec = Embedding(n_items, l_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    #item_vec = Dropout(0.35)(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    #user_bias = Dropout(0.35)(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    #item_bias = Dropout(0.35)(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    #r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer=opt)
    return model

def nn_model(n_users, n_items, l_dim):
    user_input = Input(shape = [1], name='user_input')
    item_input = Input(shape = [1], name='item_input')
    user_vec = Embedding(n_users, l_dim, embeddings_initializer='random_normal')(user_input)
    #user_vec = Dropout(0.25)(user_vec)
    user_vec = Flatten()(user_vec)
    user_hid = Dense(128, activation='relu')(user_vec)
    #user_hid = Dropout(0.25)(user_hid)
    
    item_vec = Embedding(n_items, l_dim, embeddings_initializer='random_normal')(item_input)
    #item_vec = Dropout(0.25)(item_vec)
    item_vec = Flatten()(item_vec)
    item_hid = Dense(128, activation='relu')(item_vec)
    #item_hid = Dropout(0.25)(item_hid)
    
    #merge_vec = Dot(axes=1)([user_hid, item_hid])
    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dense(168, activation='relu')(merge_vec)
    hidden = Dropout(0.75)(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    hidden = Dropout(0.75)(hidden)
    '''
    hidden = Dense(64, activation='relu')(hidden)
    hidden = Dropout(0.75)(hidden)
    hidden = Dense(32, activation='relu')(hidden)
    hidden = Dropout(0.75)(hidden)
    hidden = Dense(32, activation='relu')(hidden)
    hidden = Dropout(0.75)(hidden)
    '''
    output = Dense(1)(hidden)
    model = Model([user_input, item_input], output)
    model.compile(loss='mse', optimizer=opt)
    return model

data_path = 'data/train.csv'
users, items, rates, n_users, n_items, users_t, items_t, rates_t = load_data(data_path)
print(n_users, n_items)
model = mf_model(n_users,n_items,333)
model.summary()

filepath="check_point/mf-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


train_history = model.fit({'user_input':users, 'item_input':items}, rates, validation_split=0.01, epochs=40, batch_size=4096, callbacks=callbacks_list, shuffle =True, verbose=1)
#train_history = model.fit([users, items], rates, validation_split=0.2, epochs=30, batch_size=1024, callbacks=callbacks_list, shuffle =True, verbose=1)
train_loss = train_history.history['loss']
val_loss = train_history.history['val_loss']



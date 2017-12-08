from keras.models import Sequential, Model, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Activation, Input, LSTM, Dropout, Bidirectional
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from math import log, floor
import random
import sys
# import modules & set up logging
import gensim



def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def split_unlabeled_set(X_all, length):
    pivot =  random.randint(0,len(X_all) - length)
    return  X_all[pivot : pivot+length]


word2vec = gensim.models.Word2Vec.load('embedding')

max_len = 40 

labels = []
trainLabeled = []
trainUnlabeled = []
tests = []

labeled_data_path = 'data/training_label.txt'
text = open(labeled_data_path, 'r')
rows = text.readlines()

lens = []
for row in rows:
    labels.append(int(row[0]))
    trainLabeled.append(row[10:-1])
    trainLabeled[-1] = text_to_word_sequence(trainLabeled[-1])
    lens.append(len(trainLabeled[-1]))
    for idx in range(len(trainLabeled[-1])):
        trainLabeled[-1][idx] = word2vec[trainLabeled[-1][idx]]

print("length :", max(lens), np.mean(lens))

trainLabeled = pad_sequences(trainLabeled, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)
#trainUnlabeled = pad_sequences(trainUnlabeled, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)
labels = np.array(labels)
labels = to_categorical(labels)
print(labels.shape)

model = Sequential()
#model.add( Bidirectional( deepLSTM, input_shape=(max_len,256) ) )
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.1, return_sequences=True, input_shape=(max_len,256)))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.1, return_sequences=True))
#model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.1, return_sequences=False))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(2, activation='softmax'))

#model = load_model(sys.argv[1])

model.summary()

#opt = Adam(lr=0.001, decay = 0.01)
opt = Adadelta(lr=0.8, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#set check point
filepath="check_point/pretrain-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

loss = []
val_loss = []

train_X = pad_sequences( trainLabeled, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)
train_history = model.fit(train_X, labels, validation_split=0.1, epochs=100, batch_size=1800, callbacks=callbacks_list, shuffle =True, verbose=1)
loss = loss + train_history.history['acc']
val_loss = val_loss + train_history.history['val_acc']

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['acc', 'val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('acc.png')
plt.show()


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
from gensim import corpora

labels = []
trainLabeled = []
trainUnlabeled = []
tests = []

labeled_data_path = '../data/training_label.txt'
text = open(labeled_data_path, 'r')
rows = text.readlines()

lens = []
for row in rows:
    labels.append(int(row[0]))
    trainLabeled.append(row[10:-1])
    trainLabeled[-1] = text_to_word_sequence(trainLabeled[-1])
    lens.append(len(trainLabeled[-1]))

print("length :", max(lens), np.mean(lens))

stoplist = set('for a of the and to in'.split())
trainLabeled = [[word for word in document if word not in stoplist]
                for document in trainLabeled]
from collections import defaultdict
frequency = defaultdict(int)
for text in trainLabeled:
    for token in text:
        frequency[token] += 1

trainLabeled = [[token for token in text if frequency[token] > 25] for text in trainLabeled]

labels = np.array(labels)
labels = to_categorical(labels)
print(labels.shape)

dictionary = corpora.Dictionary(trainLabeled)
dictionary.save('corp.dict')  # store the dictionary, for future reference
print(dictionary)
print(len(dictionary))

bows = np.zeros( (len(trainLabeled),len(dictionary)) )

for i in range(len(trainLabeled)):
    bow = dictionary.doc2bow(trainLabeled[i])
    for it in bow:
        bows[i,it[0]] = it[1]

model = Sequential()
model.add(Dense(1024,input_dim = bows.shape[1], activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu'))#, input_dim = bows.shape[1]))
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

train_X = bows

train_history = model.fit(train_X, labels, validation_split=0.1, epochs=15, batch_size=1800, callbacks=callbacks_list, shuffle =True, verbose=1)
loss = loss + train_history.history['acc']
val_loss = val_loss + train_history.history['val_acc']

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['acc', 'val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('acc.png')
plt.show()


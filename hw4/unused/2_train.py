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

lens = []
unlabeled_data_path = 'data/training_nolabel.txt'
text = open(unlabeled_data_path, 'r')
rows = text.readlines()

for row in rows:
    trainUnlabeled.append(row[:-1])
    trainUnlabeled[-1] = text_to_word_sequence(trainUnlabeled[-1])
    lens.append(len(trainUnlabeled[-1]))
    #for idx in range(len(trainUnlabeled[-1])):
    #    trainUnlabeled[-1][idx] = word2vec[trainUnlabeled[-1][idx]]
    
print("length :", max(lens), np.mean(lens))

utrain_X = split_unlabeled_set(trainUnlabeled, 5 ) 
lens = []
test_data_path = 'data/testing_data.txt'
text = open(test_data_path, 'r')
rows = text.readlines()

for i,row in enumerate(rows):
    if i == 0:
        continue
    for pivot in range(len(row)):
        if row[pivot] == ',':
            tests.append(row[pivot+1:-1])
            tests[-1] = text_to_word_sequence(tests[-1])
            break
    lens.append(len(tests[-1]))
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
train_history = model.fit(train_X, labels, validation_split=0.1, epochs=40, batch_size=1800, callbacks=callbacks_list, shuffle =True, verbose=1)
loss = loss + train_history.history['acc']
val_loss = val_loss + train_history.history['val_acc']

filepath="check_point/semi-surpurvised-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

total_epochs = 0
best_acc = -0.99

for ite in range(30):
    train_X, train_Y, valid_X, valid_Y = split_valid_set(trainLabeled, labels, 0.1)
    utrain_X = split_unlabeled_set(trainUnlabeled, 2*len(train_X) )
    rounds = len(train_X) // 600
    for epochs in range(2):
        total_epochs += 1
        for rnd in range(rounds):
            semi_X = utrain_X[rnd*1200:(rnd+1)*1200]
            tX = train_X[rnd*600:(rnd+1)*600,:,:]
            tY = train_Y[rnd*600:(rnd+1)*600,:]

            for i in range(len(semi_X)):
                for idx in range(len(semi_X[i])):
                    if(type(semi_X[i][idx]) is not str):
                        continue
                    if semi_X[i][idx] in word2vec.wv.vocab:
                        semi_X[i][idx] = word2vec[semi_X[i][idx]]
                    else:
                        semi_X[i][idx] = np.zeros((256,), dtype=float)
    
            semi_X = pad_sequences(semi_X , maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)
            result = model.predict(semi_X, batch_size = 1800, verbose=0)
            threshold = 0.10
            label = np.squeeze(result[:,1])
            index = (label>1-threshold) + (label<threshold)
            semi_Y = np.greater(label, 0.5).astype(np.int32)
            semi_X = semi_X[index,:,:]
            semi_Y = to_categorical(semi_Y[index],num_classes=2)
            semi_X = np.concatenate((semi_X, tX))
            semi_Y = np.concatenate((semi_Y, tY))
            model.train_on_batch(semi_X, semi_Y)
            print('train on batch: ',rnd,'/',rounds,'\r',end="")
        metrics = model.evaluate(valid_X, valid_Y, batch_size=1800)
        print( '\nEpoch: ', total_epochs,'/','60, loss: ',metrics[0], 'acc: ',metrics[1]*100)
        if metrics[1]>best_acc:
            best_acc = metrics[1]
            model.save('check_point/semi_'+str(round(metrics[1]*100,0))+'_'+str(total_epochs)+'.hdf5')
        val_loss.append(metrics[1]*100)
    
plt.plot(val_loss)
plt.legend(['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('acc_semi.png')
plt.show()


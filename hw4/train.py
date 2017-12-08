from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Activation, Input, LSTM, Dropout, Bidirectional, Merge
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# import modules & set up logging
import gensim

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
print("length :", max(lens), np.mean(lens))

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
labels = np.array(labels)
labels = to_categorical(labels)

deepLSTM = Sequential()
deepLSTM.add(LSTM(160, dropout=0.45, recurrent_dropout=0.2, return_sequences=True, input_shape=(max_len,256)))
deepLSTM.add(LSTM(128, dropout=0.45, recurrent_dropout=0.2, return_sequences=True))
deepLSTM.add(LSTM(64, dropout=0.45, recurrent_dropout=0.2, return_sequences=False))

deepLSTM_b = Sequential()
deepLSTM_b.add(LSTM(160, dropout=0.45, recurrent_dropout=0.2, return_sequences=True, go_backwards=True, input_shape=(max_len,256)))
deepLSTM_b.add(LSTM(128, dropout=0.45, recurrent_dropout=0.2, return_sequences=True, go_backwards=True))
deepLSTM_b.add(LSTM(64, dropout=0.45, recurrent_dropout=0.2, return_sequences=False, go_backwards=True))

print(trainLabeled.shape)
print(labels.shape)
model = Sequential()
model.add( Merge([deepLSTM, deepLSTM_b], mode='concat') )
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(2, activation='softmax'))

model.summary()

opt = Adam(lr=0.005, decay= 0.06)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#set check point
filepath="check_point/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_history = model.fit([trainLabeled,trainLabeled], labels, validation_split=0.1, epochs=100, batch_size=1000, callbacks=callbacks_list, shuffle =True, verbose=1)
loss = train_history.history['acc']
val_loss = train_history.history['val_acc']

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['acc', 'val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('acc.png')
plt.show()


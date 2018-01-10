import pickle
import numpy as np
import csv
import sys
import random
import keras
import jieba
import re
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout,Input,LSTM, Concatenate
from keras.utils.np_utils import to_categorical

jieba.set_dictionary('dict.txt.big')

train_audio = pickle.load(open("data/train.data", "rb")) 
test_audio = pickle.load(open("data/test.data", "rb"))

train_ap=np.zeros((6*len(train_audio),246,39))
for i in range(len(train_audio)):
    train_ap[i                      ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    train_ap[i+1*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    train_ap[i+2*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    train_ap[i+3*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    train_ap[i+4*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    train_ap[i+5*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    #train_ap[i+6*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
    #train_ap[i+7*len(train_audio)   ,246-len(train_audio[i]):246    ,:]=train_audio[i]
# input data
train_text=[]
f=open('data/train.caption','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    s = re.sub('[\n ]', '', i)          #remove space and \n
    seg_list = jieba.cut(s, cut_all=False)
    #train_text.append(" ".join(seg_list))
    train_text.append(i)
f.close()
test_text=[]
f=open('data/test.csv','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    a=i.split(',')
    for j in a:
        s = re.sub('[\n ]', '', j)          #remove space and \n
        seg_list = jieba.cut(s, cut_all=False)
        #test_text.append(" ".join(seg_list))
        test_text.append(j)
f.close() 

t = Tokenizer()
t.fit_on_texts(train_text+test_text)
vocab_size = len(t.word_index) + 1
train_x=train_text
train_x=train_x+train_text[15:len(train_text)]+train_text[0:15]
train_x=train_x+train_text[20:len(train_text)]+train_text[0:20]
train_x=train_x+train_text[25:len(train_text)]+train_text[0:25]
train_x=train_x+train_text[30:len(train_text)]+train_text[0:30]
train_x=train_x+train_text[35:len(train_text)]+train_text[0:35]
#train_x=train_x+train_text[40:len(train_text)]+train_text[0:40]
#train_x=train_x+train_text[45:len(train_text)]+train_text[0:45]

encoded_docs = t.texts_to_sequences(train_x)
#print("train len: ", max([len(a) for a in encoded_docs]))
padded_docs = pad_sequences(encoded_docs, 15, padding='pre')
#ptext=padded_docs.reshape(2*len(train_text),15,1)
ans=[1]*len(train_text)+[0]*5*len(train_text)
ans=np.array(ans)
#ans = to_categorical(ans)
ans=ans.astype('float')

shuf=(list(range(len(ans))))
random.shuffle(shuf)

train_ap=train_ap[shuf,:,:]
padded_docs=padded_docs[shuf,:]
ans=ans[shuf]


_t = Input(shape=[15])
emb_t=Embedding(vocab_size,400,embeddings_initializer='random_normal',input_length=15)(_t)
#emb_t = Dropout(0.5)(emb_t)
_a = Input(shape=(246, 39))

encoded_t = LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(emb_t)
encoded_t = LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_t)
#encoded_t = LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_t)

encoded_a = LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(_a)
encoded_a = LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_a)
#encoded_a = LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_a)

#encoded_t = t_lstm(emb_t)
#encoded_t=Dropout(0.2)(encoded_t)

#encoded_a = a_lstm(_a)
#encoded_a=Dropout(0.2)(encoded_a)

merged_vector = keras.layers.dot([encoded_t, encoded_a],axes=1)
'''
merged_vector = keras.layers.Concatenate()([encoded_t, encoded_a])
o = Dense(256, activation='relu')(merged_vector)
o = Dropout(0.3)(o)
o = Dense(64, activation='relu')(o)
o = Dropout(0.7)(o)
'''
o = Dense(1, activation='sigmoid')(merged_vector)
model = Model(inputs=[_a, _t], outputs=o)
opt = RMSprop(lr=0.001)
model.summary()
model.compile(optimizer=opt,
              #loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#filepath="check_point/model-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="check_point/model-{epoch:02d}.hdf5"
chechpoint=keras.callbacks.ModelCheckpoint(
    filepath, 
    #monitor='val_acc', 
    verbose=1, 
    save_best_only=False, 
    #mode='max', 
    period=10
)
model.fit([train_ap, padded_docs], ans, validation_split=0.0, batch_size=512, epochs=100, callbacks=[chechpoint],verbose=1)
#model.fit([train_ap, padded_docs], ans, validation_split=0.2, epochs=30, callbacks=[chechpoint],verbose=1)

model.save('m_dlstm_1:5_ep_60_full.hdf5')
'''
# test
encoded_docs_test = t.texts_to_sequences(test_text)
print("test len: ", max([len(a) for a in encoded_docs_test]))
padded_docs_test = pad_sequences(encoded_docs_test, 15, padding='pre')
encoded_docs_test=np.array(encoded_docs_test)
test4text=padded_docs_test.reshape(len(padded_docs_test),15)
text4audio=np.zeros((len(test_text),246,39))
for i in range(2000):
    text4audio[4*i      ,246-len(test_audio[i]):246, :]=test_audio[i]
    text4audio[4*i+1    ,246-len(test_audio[i]):246, :]=test_audio[i]
    text4audio[4*i+2    ,246-len(test_audio[i]):246, :]=test_audio[i]
    text4audio[4*i+3    ,246-len(test_audio[i]):246, :]=test_audio[i]   

pre=model.predict([text4audio,test4text])
fi=[]
for i in range(2000):
    can=pre[4*i:4*i+4]
    fi.append(np.argmax(can))

f = open('final216_200_test1.txt',"w")
w = csv.writer(f)
w.writerow(('id','answer'))
for i in range(2000):
    w.writerow((i+1,fi[i]))
f.close()
'''

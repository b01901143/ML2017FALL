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
from keras.models import Model,Sequential,load_model
from keras.layers import Dense, Dropout,Input,LSTM,GRU, Concatenate
from keras.utils.np_utils import to_categorical


def downsampling(audio):
    downudio=np.zeros((len(audio),246,39))
    for i in range(len(audio)):
        daudio=audio[i,:len(audio[i]):2,:]
        downudio[i,246-len(daudio):246,:]=daudio
    return downudio

train_audio = pickle.load(open("data/train.data", "rb")) 
test_audio = pickle.load(open("data/test.data", "rb"))

# input text
train_text=[]
f=open('data/train.caption','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    s = re.sub('[\n ]', '', i)          #remove space and \n
    train_text.append(i)
f.close()
test_text=[]
f=open('data/test.csv','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    a=i.split(',')
    for j in a:
        s = re.sub('[\n ]', '', j)          #remove space and \n
        test_text.append(j)
f.close() 

#token of text
t = Tokenizer()
t.fit_on_texts(train_text+test_text)
vocab_size = len(t.word_index) + 1

'''
#model
_t = Input(shape=[15])
emb_t=Embedding(vocab_size,400,embeddings_initializer='random_normal',input_length=15)(_t)
#emb_t = Dropout(0.5)(emb_t)
_a = Input(shape=(246, 39))

encoded_t = GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(emb_t)
encoded_t = GRU(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_t)
#encoded_t = LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_t)

encoded_a = GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(_a)
encoded_a = GRU(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_a)
#encoded_a = LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(encoded_a)

#encoded_t = t_lstm(emb_t)
#encoded_t=Dropout(0.2)(encoded_t)

#encoded_a = a_lstm(_a)
#encoded_a=Dropout(0.2)(encoded_a)

merged_vector = keras.layers.dot([encoded_t, encoded_a],axes=1)
o = Dense(1, activation='sigmoid')(merged_vector)
model = Model(inputs=[_a, _t], outputs=o)
opt = RMSprop(lr=0.001)
model.summary()
model.compile(optimizer=opt,
              #loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])
'''
model = load_model(sys.argv[1])

filepath="check_point/model-{epoch:02d}+40.hdf5"
chechpoint=keras.callbacks.ModelCheckpoint(
    filepath, 
    verbose=1, 
    period=10
)

for epoch in range(10):

    #preprocessing

    #input:test_audio output:3d nparray(padding zero)
    train_ap0=np.zeros((len(train_audio),246,39))
    for i in range(len(train_audio)):
        train_ap0[i,246-len(train_audio[i]):246,:]=train_audio[i]
    
    train_down=downsampling(train_ap0)

    train_x=train_text+train_text
    ans=[]
    rm1=np.random.randint(45036, size=(45036*5,2))
    #audio_original
    for i in range((45036*5)):
        train_x.append(train_text[rm1[i,0]])
        if(train_text[rm1[i,0]]==train_text[rm1[i,1]]):
            ans.append(1)
        else:
            ans.append(0)
    #audio_down_sample
    rm2=np.random.randint(45036, size=(45036*5,2))
    for i in range((45036*5)):
        train_x.append(train_text[rm2[i,0]])
        if(train_text[rm2[i,0]]==train_text[rm2[i,1]]):
            ans.append(1)
        else:
            ans.append(0)
    ans=[1]*2*len(train_text)+ans
    ans=np.array(ans)
    #ans = to_categorical(ans)
    ans=ans.astype('float')
    train_ap=np.concatenate((train_down,train_ap0,train_down[rm1[:,1]],train_ap0[rm2[:,1]]),axis=0)


    encoded_docs = t.texts_to_sequences(train_x)
    #print("train len: ", max([len(a) for a in encoded_docs]))
    padded_docs = pad_sequences(encoded_docs, 15, padding='pre')
    #ptext=padded_docs.reshape(2*len(train_text),15,1)

    shuf=(list(range(len(ans))))
    random.shuffle(shuf)
    train_ap=train_ap[shuf,:,:]
    padded_docs=padded_docs[shuf,:]
    ans=ans[shuf]
    model.fit([train_ap, padded_docs], ans, validation_split=0.0, batch_size=512, epochs=5, verbose=1)
    model.save('check_point/model_'+str(5*epoch+5)+'+50.hdf5')

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

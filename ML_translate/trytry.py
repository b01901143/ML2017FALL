import pickle
import numpy as np
import csv
import sys
import random
import keras
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout,Input,LSTM

train_text=[]
f=open('train.caption','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    train_text.append(i)
f.close()

test_text=[]
f=open('test.csv','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    a=i.split(',')
    for j in a:
        test_text.append(j)
f.close() 

test_audio = pickle.load(open("test.data", "rb"))

model128=keras.models.load_model('128_3train_best.hdf5')

t = Tokenizer()
t.fit_on_texts(train_text+test_text)
vocab_size = len(t.word_index) + 1

encoded_docs_test = t.texts_to_sequences(test_text)
padded_docs_test = pad_sequences(encoded_docs_test, 15, padding='pre')
encoded_docs_test=np.array(encoded_docs_test)
test4text=padded_docs_test.reshape(len(padded_docs_test),15)
text4audio=np.zeros((len(test_text),246,39))
for i in range(2000):
    text4audio[4*i,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+1,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+2,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+3,246-len(test_audio[i]):246,:]=test_audio[i]   

model1=keras.models.load_model('01.hdf5')
pre1=model1.predict([text4audio,padded_docs_test])

model2=keras.models.load_model('02.hdf5')
pre2=model2.predict([text4audio,padded_docs_test])
'''
fi=[]
for i in range(int(len(pre2)/4)):
    can=pre2[4*i:4*i+4]
    fi.append(np.argmax(can))

     
f = open('final217.txt',"w")
w = csv.writer(f)
w.writerow(('id','answer'))
for i in range(2000):
    w.writerow((i+1,fi[i]))
f.close()
'''
model3=keras.models.load_model('03.hdf5')
pre3=model3.predict([text4audio,padded_docs_test])


model4=keras.models.load_model('04.hdf5')
pre4=model4.predict([text4audio,padded_docs_test])


model5=keras.models.load_model('05.hdf5')
pre5=model5.predict([text4audio,padded_docs_test])

prp6=pre1+pre2+pre3+pre4+pre5

model200=keras.models.load_model('aasd_200_1221.hdf5')
pre200=model200.predict([text4audio,padded_docs_test])
pre200=pre200+prp6
for i in range(int(len(pre5)/4)):
    can=pre200[4*i:4*i+4]
    fi.append(np.argmax(can))

  
f = open('final226.txt',"w")
w = csv.writer(f)
w.writerow(('id','answer'))
for i in range(2000):
    w.writerow((i+1,fi[i]))
f.close()
'''
26是把所有ensemble
'''

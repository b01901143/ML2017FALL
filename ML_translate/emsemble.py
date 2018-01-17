import pickle
import numpy as np
import csv
import sys
import random
import keras
import jieba
import re
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout,Input,LSTM
#model_list = ['m_dlstm_1:3.hdf5','m_dlstm_1:5_full.hdf5','m_dlstm_1:5.hdf5','m_dlstm_2:20_full_60.hdf5','m_dlstm_2:20.hdf5','m_dlstm_rand_70.hdf5','m_down_sample_60.hdf5']
model_list = ['check_point/model_15+50.hdf5','check_point/model_20+50.hdf5','check_point/model_25+50.hdf5']
model_list += ['check_point/model_35+50.hdf5','check_point/model_30+50.hdf5']
train_text=[]
f=open('data/train.caption','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    s = re.sub('[\n ]', '', i)          #remove space and \n
    seg_list = jieba.cut(s, cut_all=False)
    train_text.append(i)
    #train_text.append(" ".join(seg_list))
f.close()

test_text=[]
f=open('data/test.csv','r',encoding = 'UTF-8')
data=f.readlines()
for i in data:
    a=i.split(',')
    for j in a:
        s = re.sub('[\n ]', '', j)          #remove space and \n
        seg_list = jieba.cut(s, cut_all=False)
        test_text.append(j)
        #test_text.append(" ".join(seg_list))
f.close() 

test_audio = pickle.load(open("data/test.data", "rb"))

t = Tokenizer()
t.fit_on_texts(train_text+test_text)
vocab_size = len(t.word_index) + 1

encoded_docs_test = t.texts_to_sequences(test_text)
padded_docs_test = pad_sequences(encoded_docs_test, 15, padding='pre')
encoded_docs_test=np.array(encoded_docs_test)
test4text=padded_docs_test.reshape(len(padded_docs_test),15)
text4audio=np.zeros((len(test_text),246,39))

def downsampling(audio):
    downudio=np.zeros((len(audio),246,39))
    for i in range(len(audio)):
        daudio=audio[i,:len(audio[i]):2,:]
        downudio[i,246-len(daudio):246,:]=daudio
    return downudio

for i in range(2000):
    text4audio[4*i,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+1,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+2,246-len(test_audio[i]):246,:]=test_audio[i]
    text4audio[4*i+3,246-len(test_audio[i]):246,:]=test_audio[i]   

test_down=downsampling(text4audio)
predictions = []
predictions_d = []
for a in model_list:
    model=keras.models.load_model(a)
    pre=model.predict([text4audio,padded_docs_test], batch_size=512)
    predictions.append(pre)
    pre_down=model.predict([test_down,padded_docs_test], batch_size=512)
    predictions_d.append(pre_down)
fi=[]
for i in range(int(len(predictions[0])/4)):
    prob = np.ones((4,1), dtype=np.float64)
    for j in range(len(predictions)):
        prob *= predictions[j][4*i : 4*i+4]
        prob *= predictions_d[j][4*i : 4*i+4]
    fi.append(np.argmax(prob))

  
f = open('out_emsemble.txt',"w")
w = csv.writer(f)
w.writerow(('id','answer'))
for i in range(2000):
    w.writerow((i+1,fi[i]))
f.close()
'''
26是把所有ensemble
'''

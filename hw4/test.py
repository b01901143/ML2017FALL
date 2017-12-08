import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, LSTM
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

output_path = sys.argv[2]

# import modules & set up logging
import gensim

word2vec = gensim.models.Word2Vec.load('embedding')

max_len = 40 

tests = []

test_data_path = sys.argv[1]
text = open(test_data_path, 'r')
rows = text.readlines()

for i,row in enumerate(rows):
    if i == 0:
        continue
    for pivot in range(len(row)):
        if row[pivot] == ',':
            tests.append(row[pivot+1:-1])
            tests[-1] = text_to_word_sequence(tests[-1], filters='\t\n')
            for idx in range(len(tests[-1])):
                tests[-1][idx] = word2vec[tests[-1][idx]]
            break

tests = pad_sequences(tests, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)

model = load_model( sys.argv[3])
result = model.predict(tests, batch_size = 1000)
y_ = np.argmax(result, axis=1)
print('=====Write output to %s =====' % output_path)
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_):
        f.write('%d,%d\n' %(i, v))

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

labels = []
trainLabeled = []
trainUnlabeled = []
tests = []

labeled_data_path = '../data/training_label.txt'
text = open(labeled_data_path, 'r')
rows = text.readlines()

for row in rows:
    labels.append(int(row[0]))
    trainLabeled.append(row[10:-1])
    trainLabeled[-1] = text_to_word_sequence(trainLabeled[-1], filters='\t\n')

unlabeled_data_path = '../data/training_nolabel.txt'
text = open(unlabeled_data_path, 'r')
rows = text.readlines()

for row in rows:
    trainUnlabeled.append(row[:-1])
    trainUnlabeled[-1] = text_to_word_sequence(trainUnlabeled[-1], filters='\t\n')

test_data_path = '../data/testing_data.txt'
text = open(test_data_path, 'r')
rows = text.readlines()

for i,row in enumerate(rows):
    if i == 0:
        continue
    for pivot in range(len(row)):
        if row[pivot] == ',':
            tests.append(row[pivot+1:-1])
            tests[-1] = text_to_word_sequence(tests[-1], filters='\t\n')
            break

sentences = trainLabeled + trainUnlabeled + tests

model = gensim.models.Word2Vec(sentences, min_count=1, size=256, sg=0, workers=32, iter=10)

model.save('embedding')

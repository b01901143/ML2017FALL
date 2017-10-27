import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import load_model

# If you wish to get the same shuffle result
# np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)


    # X_train = np.delete(X_train, [a for a in range(0,2)], 1)
    # X_test = np.delete(X_test, [a for a in range(0,2)], 1)


    # X_train = np.delete(X_train, [a for a in range(15,22)], 1)
    # X_test = np.delete(X_test, [a for a in range(15,22)], 1)

    # edu =  pd.read_csv("data/train.csv", sep=',', header=0)
    # edu = np.reshape(np.array(edu.values)[:,4].astype(int),[edu.shape[0],1])

    # X_train = np.hstack((X_train,edu))
    # edu = edu**2
    # X_train = np.hstack((X_train,edu))

    # edu =  pd.read_csv("data/test.csv", sep=',', header=0)
    # edu = np.reshape(np.array(edu.values)[:,4].astype(int),[edu.shape[0],1])

    # X_test = np.hstack((X_test,edu))
    # edu = edu**2
    # X_test = np.hstack((X_test,edu))
    
    return (X_train, Y_train, X_test)


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    # cross_entropy = -1 * (np.dot(np.squeeze(Y_valid), np.log(y)) + np.dot((1 - np.squeeze(Y_valid)), np.log(1 - y)))
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    # print('epoch avg loss = %f' % (float(cross_entropy) / valid_data_size) )
    return

def train(X_all, Y_all, save_dir, _valid):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1

    # Y_tmp = []
    # for i in range(Y_all.shape[0]):
    #     if Y_all[i] == 1:
    #         Y_tmp.append([1,0])
    #     else:
    #         Y_tmp.append([0,1])
    # Y_all = np.array(Y_tmp)

    # Y_all = keras.utils.to_categorical(Y_all, num_classes = 2)

    if _valid:
        X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    else:
        X_train, Y_train= X_all, Y_all

    model = Sequential()
    layer_size = X_train.shape[1] // 2
    model.add(Dense(input_dim=X_train.shape[1], units=32, activation = 'sigmoid'))
    # model.add(Dropout(0.3))
    for i in range(1):
        layer_size = layer_size // 2
        model.add(Dense(units=32, activation = 'sigmoid'))
        model.add(Dropout(0.8))
   
    model.add(Dense(units=1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=128, epochs = 100)

    if _valid:
        score = model.evaluate(X_valid, Y_valid)
        print("\nValidation loss: ", score[0])
        print("\nValidation accuracy: ", score[1])

    # print(X_train.shape)

    return model

def infer(X_test, save_dir, output_path):
    model = load_model('nn_model.h5')
    result = model.predict(X_test)
    y_ = [] 
    for i in result:
        if i[0] >0.5:
            y_.append(1)
        else:
            y_.append(0)
    print('=====Write output to %s =====' % opts.output_path)
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))


    return

def main(opts):
    # Load feature and label
    X_all, Y_all, X_test = load_data(opts.train_data_path, opts.train_label_path, opts.test_data_path)

    # X_all, Y_all, X_test, _ = parse_data()

    # Normalization
    X_all, X_test = normalize(X_all, X_test)

    # To train or to infer
    if opts.train:
        model = train(X_all, Y_all, opts.save_dir, opts.valid)
        model.save('nn_model.h5')
    elif opts.infer:
        infer(X_test, opts.save_dir, opts.output_path)
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--valid', action='store_true', default=False,
                        dest='valid', help='Input --valid to Cut Validation')
    parser.add_argument('--train_data_path', type=str,
                        default='feature/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='feature/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='feature/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_dir', type=str,
                        default='nn_params/', dest='save_dir',
                        help='Path to save the model parameters')
    parser.add_argument('--output_path', type=str,
                        default='nn_output/', dest='output_path',
                        help='Path to save the model parameters')
    opts = parser.parse_args()
    main(opts)

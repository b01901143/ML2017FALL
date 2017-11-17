import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
import sys
import os
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

print(device_lib.list_local_devices())

def load_data(train_data_path):
    X_train = []
    Y_train = []

    text = open(train_data_path, 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for i,r in enumerate(row):
        if i == 0:
            continue
        Y_train.append(int(r[0]))
        X_train.append(r[1].split())



    return ( np.reshape(np.array(X_train,dtype='int'),(len(X_train),48*48)), keras.utils.to_categorical(np.array(Y_train,dtype='int')) )


if __name__ == '__main__':
    train_data_path = 'data/train.csv'
    X_train, Y_train = load_data(train_data_path)
    num_classes = Y_train.shape[1]

    print(X_train.shape)
    print(Y_train.shape)

    
    #model = load_model('check_point/'+sys.argv[1])
    
    model = Sequential()

    model.add(Dense(480, activation='relu', input_dim = 48*48))
    model.add(Dropout(.5))
    model.add(Dense(240, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(70, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(7, activation='softmax'))
    model.summary()

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-6)
    opt = Adadelta(lr=0.5, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #set check point
    filepath="check_point/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    '''
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),steps_per_epoch=len(X_train) / 32, epochs=300)
    
    for e in range(300):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    '''
    train_history = model.fit(X_train, Y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=callbacks_list, shuffle =True, verbose=1)
    loss = train_history.history['acc']
    val_loss = train_history.history['val_acc']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['acc', 'val_acc'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('acc_nn.png')
    plt.show()

    model.summary()
    plot_model(model,to_file='model_nn.png')

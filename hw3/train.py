import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
import os

def load_data(train_data_path):
    X_train = []
    Y_train = []

    text = open('data/train.csv', 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for i,r in enumerate(row):
        if i == 0:
            continue
        Y_train.append(int(r[0]))
        X_train.append(r[1].split())



    return ( np.reshape(np.array(X_train),(len(X_train),48,48,1)), keras.utils.to_categorical(np.array(Y_train)) )


if __name__ == '__main__':
    train_data_path = 'data/train.csv'
    X_train, Y_train = load_data(train_data_path)
    num_classes = Y_train.shape[1]

    print(X_train.shape)
    print(Y_train.shape)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    #set check point
    filepath="check_point/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(X_train, Y_train, validation_split=0.2, epochs=150, batch_size=32, callbacks=callbacks_list, verbose=0)


import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
import sys
import os
from tensorflow.python.client import device_lib
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



    return ( np.reshape(np.array(X_train,dtype='int'),(len(X_train),48,48,1)), keras.utils.to_categorical(np.array(Y_train,dtype='int')) )


if __name__ == '__main__':
    train_data_path = 'data/train.csv'
    X_train, Y_train = load_data(train_data_path)
    num_classes = Y_train.shape[1]

    print(X_train.shape)
    print(Y_train.shape)
    
    #model = load_model('check_point/'+sys.argv[1])
     
    input_img = Input(shape=(48, 48, 1))
    block1 = Conv2D(50, (3, 3), padding='same', activation='relu')(input_img)
    block1 = Dropout(0.2)(block1)

    block2 = Conv2D(50, (5, 5), padding='same', activation='relu')(block1)
    block2 = AveragePooling2D(pool_size=(4, 4), strides=(2, 2))(block2)
    block2 = Dropout(0.2)(block2)

    block3 = Conv2D(50, (5, 5), activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block3)
    block3 = Dropout(0.2)(block3)

    fc1 = Flatten()(block3)
    fc1 = Dense(200, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(100, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    #set check point
    filepath="check_point/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(X_train, Y_train, validation_split=0.2, epochs=1500, batch_size=256, callbacks=callbacks_list, shuffle =True, verbose=1)

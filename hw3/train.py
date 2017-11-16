import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
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
    block1 = Conv2D(128, (5, 5), activation='relu')(input_img)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(block1)
    block1 = Dropout(0.25)(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block2)
    block2 = Dropout(0.25)(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = Dropout(0.25)(block3)

    block4 = Conv2D(32, (2, 2), activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block4)
    block4 = Dropout(0.25)(block4)

    block5 = Conv2D(12, (3, 3), activation='relu')(block4)
    block5 = Dropout(0.25)(block5)

    block6 = Conv2D(16, (2, 2), activation='relu')(block5)
    block6 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block6)
    block6 = Dropout(0.25)(block6)

    #block7 = Conv2D(32, (3, 3), activation='relu')(block6)
    #block7 = Dropout(0.25)(block7)
    #block7 = BatchNormalization()(block7)

    #block8 = Conv2D(32, (2, 2), activation='relu')(block7)
    #block8 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block8)
    #block8 = Dropout(0.25)(block8)
    #block8 = BatchNormalization()(block8)

    fc1 = Flatten()(block6)
    fc1 = Dense(300, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(300, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    fc2 = BatchNormalization()(fc2)
    
    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-6)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    #set check point
    filepath="check_point/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(X_train, Y_train, validation_split=0.1, epochs=300, batch_size=32, callbacks=callbacks_list, shuffle =True, verbose=1)

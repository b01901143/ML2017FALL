from keras.models import loadmodel
from sklearn.metrics import confusionmatrix
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

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



    return ( np.reshape(np.array(X_train,dtype='int'),(len(X_train),48,48,1)), np.array(Y_train,dtype='int') )

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def plotconfusionmatrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    model_path = 'check_point/'+sys.argv[1]
    train_data_path = 'data/train.csv'
    dev_feats, te_labels = load_data(train_data_path)
    dev_feats, te_labels = split_valid_set( dev_feats, te_labels, 0.1 )

    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    predictions = emotion_classifier.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    print (predictions)
    print (te_labels)
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()

if __name=='__main':
    main()

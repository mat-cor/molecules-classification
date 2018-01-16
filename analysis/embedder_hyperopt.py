from __future__ import print_function

import os
import sys
import csv
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, jaccard_similarity_score
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score, f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense, Flatten, Activation
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras import regularizers
from keras import backend as K

from hyperopt import Trials, STATUS_OK, tpe
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    DATA_LOC = '../data/'
    with open(DATA_LOC+'termdict.pickle', 'rb') as handle:
        termdict = pickle.load(handle)
    with open(DATA_LOC+'smiles_vocabulary.pickle', 'rb') as handle:
        vocabulary = pickle.load(handle)    
    smiles = np.load(DATA_LOC+'smiles.npy')
    # t = Tokenizer(filters='', lower=False, char_level=True)
    # t.fit_on_texts(smiles)
    # seqs = t.texts_to_sequences(smiles)
    seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
    
    X = pad_sequences(seqs, padding='post')
    y = np.load(DATA_LOC+'multi_labels.npy')
    
    seed = 7
    np.random.seed(seed)
    # Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, y_train, X_test, y_test, vocabulary


def create_model(X_train, y_train, X_test, y_test, vocabulary):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    sequence_length = X_train.shape[1]
    vocabulary_size = len(vocabulary)
    n_class = y_train.shape[1]
    embedding_size = 64
    
    # Model
    model = Sequential()
    model.add(Embedding(output_dim=embedding_size, input_dim=vocabulary_size,
                        input_length=sequence_length))
    model.add(Convolution1D(32, kernel_size={{choice([2, 3, 4])}}, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Convolution1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Flatten())
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    model.fit(X_train, y_train, epochs=100, batch_size={{choice([64, 128])}}, verbose=2)
    
    y_prob = model.predict(X_test)
    # Model evaluation
    threshold = np.arange(0.1,1,0.1)
    accuracies = []
    best_threshold = np.zeros(n_class)
    for i in range(n_class):
        acc = []
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob[:, i]]
            acc.append(matthews_corrcoef(y_test[:,i], y_pred))
        acc = np.array(acc)
        index = np.where(acc==acc.max()) 
        accuracies.append(acc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        
    y_pred = np.array([[1 if y_prob[i,j] >= best_threshold[j] else 0 for j
                        in range(n_class)] for i in range(len(y_test))])
    
    # CA and average AUC
    acc = accuracy_score(y_test, y_pred)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    
    gpu = str(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    # X_train, Y_train, X_test, Y_test = data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)



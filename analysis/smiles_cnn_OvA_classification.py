import os
import sys
import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss, roc_auc_score

from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras import regularizers
from keras import backend as K


# The default Tensorflow behavior is to allocate memory on all the available GPUs, even if it runs only on the selected
# one. To avoid it, only the free GPU (defined by cmd line input)
gpu = str(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

DATA_LOC = '../data/'
with open(DATA_LOC+'termdict.pickle', 'rb') as handle:
    termdict = pickle.load(handle)
smiles = np.load(DATA_LOC+'smiles.npy')
t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)
X = pad_sequences(seqs, padding='post')
y_multi = np.load(DATA_LOC+'multi_labels.npy')

sequence_length = X.shape[1]
vocabulary_size = len(t.word_index)
n_class = y_multi.shape[1]
embedding_size = 64

seed = 7
np.random.seed(seed)

X_train, X_test, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=0.2, random_state=seed)

# Model
model = Sequential()
model.add(Embedding(output_dim=embedding_size, input_dim=vocabulary_size,
                    input_length=sequence_length))
model.add(Convolution1D(32, 2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Convolution1D(32, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

auc = []

for i in range(n_class):
    y_train = y_train_m[:, i]
    y_test = y_test_m[:, i]
    
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
    # evaluate the model
    probs = model.predict(X_test)
    auc.append(roc_auc_score(y_test, probs))
    
    print(termdict[i], auc[i])
    

with open('labels_auc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)    
    for i, auc in enumerate(auc):
        writer.writerow([termdict[i], auc])

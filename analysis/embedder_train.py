import os
import sys
import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, jaccard_similarity_score
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score, f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
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
y = np.load(DATA_LOC+'multi_labels.npy')

print('Number of examples: ', X.shape[0])
print('Multi-label classification, number of classes: ', y.shape[1])

sequence_length = X.shape[1]
vocabulary_size = len(t.word_index)
n_class = y.shape[1]
embedding_size = 64

# Model
model = Sequential()
model.add(Embedding(output_dim=embedding_size, input_dim=vocabulary_size,
                    input_length=sequence_length))
model.add(Convolution1D(32, 2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Convolution1D(32, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=100, batch_size=64, verbose=1)
model.save('fp-embedder.h5')
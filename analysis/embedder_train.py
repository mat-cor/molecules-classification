import os
import sys
import pickle
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, Dropout, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K

from preprocess.data_handler import load_data, load_pickle, categorical_labels

# The default Tensorflow behavior is to allocate memory on all the available GPUs, even if it runs only on the selected
# one. To avoid it, select by cmd line input the free GPU
gpu = str(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

DATA_LOC = '../data/'
termdict = load_pickle(DATA_LOC+'termdict.pickle')
vocabulary = load_pickle(DATA_LOC+'smiles_vocabulary.pickle')
dataset = load_data(DATA_LOC+'dataset.csv')
smiles = dataset['SMILES']

seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
X = pad_sequences(seqs, padding='post')

y = categorical_labels(dataset['Terms'], termdict)

sequence_length = X.shape[1]
vocabulary_size = len(vocabulary)
n_class = y.shape[1]
embedding_size = 64

# Model
model = Sequential()
model.add(Embedding(output_dim=embedding_size, input_dim=vocabulary_size,
                    input_length=sequence_length))
model.add(Convolution1D(32, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Convolution1D(32, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# model.fit(X, y, epochs=100, batch_size=64, verbose=1)
# model.save('fp-embedder.h5')

import os
import sys
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
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
smiles = np.load(DATA_LOC+'smiles.npy')
t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)
X = pad_sequences(seqs)
y = np.load(DATA_LOC+'multi_labels.npy')

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Number of training examples: ', X_train.shape[0])
print('Number of test examples: ', X_test.shape[0])
print('Multi-label classification, number of classes: ', y_train.shape[1])

print('1st example: \n', X_train[0])
print('Length of 1st sequence: ', len(seqs[0]))

sequence_length = X.shape[1]
vocabulary_size = len(t.word_index)
n_class = y.shape[1]
embedding_size = 16

# Model
model = Sequential()
model.add(Embedding(output_dim=embedding_size, input_dim=vocabulary_size,
                    input_length=sequence_length))
model.add(Convolution1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Convolution1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.2)))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_class, activation='sigmoid'))

# lr and decay to be optimized
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
score = model.evaluate(X_test, y_test, batch_size=32)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print('Output of embedding layer: \n')
get_0_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[0].output])
print(get_0_layer_output([X_train, 0])[0][0])

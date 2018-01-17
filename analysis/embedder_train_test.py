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


# POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned
# def weighted_binary_crossentropy(target, output):
#     """
#     Weighted binary crossentropy between an output tensor 
#     and a target tensor. POS_WEIGHT is used as a multiplier 
#     for the positive targets.

#     Combination of the following functions:
#     * keras.losses.binary_crossentropy
#     * keras.backend.tensorflow_backend.binary_crossentropy
#     * tf.nn.weighted_cross_entropy_with_logits
#     """
#     tfb = K.tensorflow_backend
#     # transform back to logits
#     _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
#     output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
#     output = tf.log(output / (1 - output))
#     # compute weighted loss
#     loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
#                                                     logits=output,
#                                                     pos_weight=POS_WEIGHT)
#     return tf.reduce_mean(loss, axis=-1)


# The default Tensorflow behavior is to allocate memory on all the available GPUs, even if it runs only on the selected
# one. To avoid it, only the free GPU (defined by cmd line input)
gpu = str(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

DATA_LOC = '../data/'
with open(DATA_LOC+'termdict_5t.pickle', 'rb') as handle:
    termdict = pickle.load(handle)
with open(DATA_LOC+'smiles_vocabulary.pickle', 'rb') as handle:
    vocabulary = pickle.load(handle)    
smiles = np.load(DATA_LOC+'smiles_5t.npy')
seqs = [[vocabulary[c] for c in list(s)] for s in smiles]

X = pad_sequences(seqs, padding='post')
y = np.load(DATA_LOC+'labels_5t.npy')

seed = 7
np.random.seed(seed)
# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
print('Number of examples: ', X_train.shape[0])
print('Multi-label classification, number of classes: ', y.shape[1])

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
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

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
ca_av = accuracy_score(y_test, y_pred)
auc_av = roc_auc_score(y_test, y_prob, average='micro')
print('Classification Accuracy: ', ca_av)
print('AUC: ', auc_av)

# AUC for each term
# aucs = roc_auc_score(y_test, y_prob, average=None)
# with open('multi_labels_auc.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Term', 'auc'])
#     for i, auc in enumerate(aucs):
#         writer.writerow([termdict[i], auc])
        
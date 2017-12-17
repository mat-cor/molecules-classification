import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras import regularizers
from keras import backend as K


POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned
def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    tfb = K.tensorflow_backend
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)



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
X = pad_sequences(seqs, padding='post')
y = np.load(DATA_LOC+'multi_labels.npy')

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Number of train examples: ', X_train.shape[0])
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
# model.add(Convolution1D(32, 3, activation='relu'))
# model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(n_class, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=300, batch_size=64, verbose=1)
out = model.predict(X_test)
out = np.array(out, dtype=np.float32)

# # Thresholdin probabilites at 0.5
# y_pred = np.zeros(out.shape)
# y_pred[np.where(out>=0.5)] = 1

# # Thresholding probabilities adapting the threshold for each label
threshold = np.arange(0.1,1,0.1)
acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])

for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_test[:,i], y_pred))
    acc = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []
y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j\
                    in range(y_test.shape[1])] for i in range(len(y_test))])

total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == n_class])

print('hamming_loss: ', hamming_loss(y_test, y_pred))
print('Acc: ', str(total_correctly_predicted/y_test.shape[0]))
print('total_correctly_predicted: ', total_correctly_predicted)


# # Visualize some true labels, probs and preds
# for x in range(10):
#     true = y_test[x]
#     pred_prob = out[x]
#     pred = y_pred[x]
#     print('\nTrue ', true[np.where(true==1)])
#     print('Prob ', pred_prob[np.where(true==1)])
#     print('Pred', pred[np.where(true==1)])
#     print('Pred', pred[np.where(pred==1)])
#     print('Prob ', pred_prob[np.where(pred==1)])
#     print('True ', true[np.where(pred==1)])


# # Top layer and "fingerprint" output
# get_fp_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                   [model.layers[-2].output])
# fp = get_fp_layer_output([X_test, 0])[0]
# print(' "Fingerprint": ')
# print(fp)
# get_top_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                  [model.layers[-1].output])
# top_out = get_top_layer_output([X_test, 0])[0]
# print(' Output: ')
# print(top_out)


# # Evaluation using MCC to determine the thresholds for each label for converting
# # probs to classes (if prob>=threshold --> label=1)


# # Summarize accuracy history
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig(DATA_LOC+'AccuracyHistory.png')

# # Summarize loss history
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig(DATA_LOC+'LossHistory.png')

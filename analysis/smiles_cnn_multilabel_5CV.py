import os
import sys
import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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
y = np.load(DATA_LOC+'multi_labels.npy')


# define 5-fold cross validation test harness
kfold = KFold(n_splits=5, shuffle=True)

sequence_length = X.shape[1]
vocabulary_size = len(t.word_index)
n_class = y.shape[1]
embedding_size = 64

average_auc, average_ca, average_f1 = [], [], []  # average metric value for each fold
labels_auc, labels_ca, labels_f1 = [], [], []  # list of metric values for each 
                                               # label, for each fold
f = 1

for train, test in kfold.split(X, y):
    print('Fold ', f)
    f += 1
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
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
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)
    # Evaluate the model
    out = model.predict(X_test)
    out = np.array(out, dtype=np.float32)
    # Thresholding probabilities adapting the threshold for each label
    threshold = np.arange(0.1,1,0.1)
    mcc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    
    for i in range(out.shape[1]):
        y_prob = np.array(out[:,i])
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob]
            mcc.append(matthews_corrcoef(y_test[:,i], y_pred))
        mcc = np.array(mcc)
        index = np.where(mcc==mcc.max()) 
        accuracies.append(mcc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        mcc = []
    
    y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j\
                        in range(y_test.shape[1])] for i in range(len(y_test))])
                        
    average_auc.append(roc_auc_score(y_test, out, average='micro'))     
    average_ca.append(accuracy_score(y_test, y_pred))


with open('multilabels_5CV.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['AUC', 'CA'])
    writer.writerow([str(round(np.mean(average_auc), 2))+'('+str(round(np.std(average_auc), 2))+')',
                    str(round(np.mean(average_ca), 2))+'('+str(round(np.std(average_ca), 2))+')'])
                    



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

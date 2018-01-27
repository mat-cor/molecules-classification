import os
import time
import numpy as np
import pickle

import deepchem as dc
from deepchem.utils.save import load_from_disk

import tensorflow as tf

from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.realpath("__file__"))

dataset_file = "tox21.csv"
dataset = load_from_disk(dataset_file)
pretty_columns = (
    "[" + ",".join(["'%s'" % column for column in dataset.columns.values]))


print("Columns of dataset: %s" % pretty_columns)
print("Number of examples in dataset: %s" % str(dataset.shape[0]))
smiles_field = 'smiles'
tasks = [col for col in dataset.columns.values[:12]]
tasks_auc_cnn = []
tasks_auc_ecfp = []

smiles = dataset[smiles_field]
# CNN embedding
with open('../data/smiles_vocabulary.pickle', 'rb') as handle:
    vocabulary = pickle.load(handle)
seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
print('Embedding smiles...')
start_time = time.time()
data = pad_sequences(seqs, padding='post', maxlen=1021)
model = load_model('../analysis/fp-embedder-46t.h5')
embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
fps_raw = embedder.predict(data, batch_size=1000)
print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps_raw.shape[0]))

for t in tasks:
    class_field = t
    fps = []
    labels = []
    for m, c in zip(fps_raw, dataset[class_field]):
        if c != '':  # exclude missing
            fps.append(m)
            labels.append(int(c))

    # 10-fold CV on CNN embedding

    # rf = RandomForestClassifier(n_estimators=500)
    logreg = LogisticRegression()
    auc_cnn = cross_val_score(logreg, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
    tasks_auc_cnn.append(auc_cnn.mean())
    # auc_rf = cross_val_score(rf, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
    # print('\n-----RF, 10-fold CV on CNN fingerprint-----')
    # print(auc_rf.mean(), auc_rf.std(), '\n')

    # 10-fold CV on ECFP fingerprint
    featurizer_func = dc.feat.CircularFingerprint(size=512)
    loader = dc.data.CSVLoader(tasks=[class_field], smiles_field=smiles_field, id_field=smiles_field,
                               featurizer=featurizer_func)
    ds = loader.featurize(dataset_file)

    X = np.array(ds.X)
    y = np.array(ds.y, dtype=np.int32)
    y = y.reshape(y.shape[0],)
    print(X.shape, y.shape)

    auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    print(t, '\n-----LogReg, 10-fold CV on CNN fingerprint-----')
    print(auc_cnn.mean(), auc_cnn.std(), '\n')
    print('-----LogReg, 10-fold CV on ECFP fingerprint-----')
    print(auc.mean(), auc.std())
    tasks_auc_ecfp.append(auc.mean())
    # auc_rf = cross_val_score(rf, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    # print('\n-----RF, 10-fold CV on ECFP fingerprint-----')
    # print(auc_rf.mean(), auc_rf.std(), '\n')

print('CNN-FP: ', np.mean(tasks_auc_cnn))
print('ECFP: ', np.mean(tasks_auc_ecfp))


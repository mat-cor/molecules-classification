import os
import time
import numpy as np
import pickle

import deepchem as dc
from deepchem.utils.save import load_from_disk

import tensorflow as tf

from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

dataset_file = "bace.csv"
dataset = load_from_disk(dataset_file)
num_display=10
pretty_columns = (
    "[" + ",".join(["'%s'" % column for column in dataset.columns.values[:num_display]])
    + ",...]")
print("Columns of dataset: %s" % pretty_columns)


smiles_field = 'mol'
class_field = 'Class'

smiles = [m for m in dataset[smiles_field]]
labels = [c for c in dataset[class_field]]

# 10-fold CV on CNN embedding
print('Embedding smiles...')
start_time = time.time()
with open('../data/smiles_vocabulary.pickle', 'rb') as handle:
    vocabulary = pickle.load(handle)
seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
data = pad_sequences(seqs, padding='post', maxlen=1021)
model = load_model('../analysis/fp-embedder.h5')
embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
fps = embedder.predict(data, batch_size=1000)
print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps.shape[0]))

#
rf = RandomForestClassifier(n_estimators=500)
logreg = LogisticRegression()
# auc_lr_cnn = cross_val_score(logreg, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
# auc_rf_cnn = cross_val_score(rf, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
#

# 10-fold CV on ECFP fingerprint
featurizer_func = dc.feat.CircularFingerprint(size=512)
loader = dc.data.CSVLoader(tasks=[class_field], smiles_field=smiles_field, id_field=smiles_field,
                           featurizer=featurizer_func)
dataset = loader.featurize(dataset_file)
X = np.array(dataset.X)
y = np.array(dataset.y, dtype=np.int32)
y = y.reshape(y.shape[0],)
print(len(dataset.ids))
print(X.shape, y.shape)
# auc_lr = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
# auc_rf = cross_val_score(rf, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
#
# print('-----LogReg, 10-fold CV on CNN fingerprint-----')
# print(auc_lr_cnn.mean(), auc_lr_cnn.std())
# print('-----RF, 10-fold CV on CNN fingerprint-----')
# print(auc_rf_cnn.mean(), auc_rf_cnn.std())
#
# print('-----LogReg, 10-fold CV on ECFP fingerprint-----')
# print(auc_lr.mean(), auc_lr.std())
# print('-----RF, 10-fold CV on ECFP fingerprint-----')
# print(auc_rf.mean(), auc_rf.std())

# Combine features, CNN-FP + ECFP
valid_inds = [i for i, s in enumerate(smiles) if s in dataset.ids]
print(len(fps[valid_inds]))
fp_combo = [np.concatenate((c, e)) for c, e in zip(fps[valid_inds], X)]

fp_combo = np.array(fp_combo)
print(fp_combo.shape)
auc_lr = cross_val_score(logreg, fp_combo, y, cv=10, scoring='roc_auc', n_jobs=-1)
auc_rf = cross_val_score(rf, fp_combo, y, cv=10, scoring='roc_auc', n_jobs=-1)
print('-----LogReg, 10-fold CV on combined fingerprints-----')
print(auc_lr.mean())
print('-----RF, 10-fold CV on combined fingerprints-----')
print(auc_rf.mean())

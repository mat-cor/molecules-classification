import os
import time
import numpy as np

import deepchem as dc
from deepchem.utils.save import load_from_disk

import tensorflow as tf

from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer
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

crystal_dataset_file = "crystal_desc_canvas_aug30.csv"

print("Columns of dataset: %s" % pretty_columns)
print("Number of examples in dataset: %s" % str(dataset.shape[0]))


# 10-fold CV on CNN embedding
smiles = [m for m in dataset['mol']]
labels = [c for c in dataset['Class']]
print('Embedding smiles...')
start_time = time.time()
t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)
data = pad_sequences(seqs, padding='post', maxlen=1021)

model = load_model('../analysis/fp-embedder-5t.h5')
embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
fps = embedder.predict(data, batch_size=1000)
print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps.shape[0]))

rf = RandomForestClassifier(n_estimators=500)
logreg = LogisticRegression()
auc = cross_val_score(logreg, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
print('\n-----LogReg, 10-fold CV on CNN fingerprint-----')
print(auc.mean(), auc.std(), '\n')
auc_rf = cross_val_score(rf, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
print('\n-----RF, 10-fold CV on CNN fingerprint-----')
print(auc_rf.mean(), auc_rf.std(), '\n')

# 10-fold CV on ECFP fingerprint
featurizer_func = dc.feat.CircularFingerprint(size=512)
loader = dc.data.CSVLoader(tasks=["Class"], smiles_field="mol", id_field="mol",
                           featurizer=featurizer_func)
dataset = loader.featurize(dataset_file)

X = np.array(dataset.X)
y = np.array(dataset.y, dtype=np.int32)
y = y.reshape(y.shape[0],)
print(X.shape, y.shape)

auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
print('\n-----LogReg, 10-fold CV on ECFP fingerprint-----')
print(auc.mean(), auc.std())
auc_rf = cross_val_score(rf, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
print('\n-----RF, 10-fold CV on ECFP fingerprint-----')
print(auc_rf.mean(), auc_rf.std(), '\n')

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

for t in tasks:
    class_field = t
    smiles = []
    labels = []
    for m, c in zip(dataset[smiles_field], dataset[class_field]):
        if c != '':  # exclude missing
            smiles.append(m)
            labels.append(int(c))

    # 10-fold CV on CNN embedding
    print('Embedding smiles...')
    start_time = time.time()
    tok = Tokenizer(filters='', lower=False, char_level=True)
    tok.fit_on_texts(smiles)
    seqs = tok.texts_to_sequences(smiles)
    data = pad_sequences(seqs, padding='post', maxlen=1021)

    model = load_model('../analysis/fp-embedder.h5')
    embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
    fps = embedder.predict(data, batch_size=1000)
    print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps.shape[0]))
    # rf = RandomForestClassifier(n_estimators=500)
    logreg = LogisticRegression()
    auc = cross_val_score(logreg, fps, labels, cv=10, scoring='roc_auc', n_jobs=-1)
    print(t, '\n-----LogReg, 10-fold CV on CNN fingerprint-----')
    print(auc.mean(), auc.std(), '\n')
    tasks_auc_cnn.append(auc.mean())
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
    print(t, '\n-----LogReg, 10-fold CV on ECFP fingerprint-----')
    print(auc.mean(), auc.std())
    tasks_auc_ecfp.append(auc.mean())
    # auc_rf = cross_val_score(rf, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    # print('\n-----RF, 10-fold CV on ECFP fingerprint-----')
    # print(auc_rf.mean(), auc_rf.std(), '\n')

print('CNN-FP: ', np.mean(tasks_auc_cnn))
print('ECFP: ', np.mean(tasks_auc_ecfp))


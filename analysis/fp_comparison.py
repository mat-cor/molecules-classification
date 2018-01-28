import os
import sys
parent_path = os.path.abspath(os.path.join('..'))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import datetime
import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from preprocess.data_handler import load_data, categorical_labels, load_pickle
from preprocess.smiles_embedder import get_cnn_fingerprint

path = '../data/'
termdict = load_pickle(path+'termdict.pickle')

dataset = load_data(path+'dataset.csv')
smiles = dataset['SMILES']
labels = categorical_labels(dataset['Terms'], termdict)

cnn_fp_data = get_cnn_fingerprint(smiles)
ecfp_d = load_pickle('../data/ecfp-data.pickle')
ecfp_l = load_pickle('../data/ecfp-labels.pickle')
ecfp_data, ecfp_labels = [], []
for cid in sorted(ecfp_d.keys()):
    ecfp_data.append(ecfp_d[cid])
    ecfp_labels.append(ecfp_l[cid])

ecfp_data = np.array(ecfp_data)
ecfp_labels = np.array(ecfp_labels)

with open('../results/LRauc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Term', 'CNNFP', 'ECFP'])

    k = 0
    for t in termdict.keys():

        k += 1
        print(k)
        print(datetime.datetime.now())

        y = labels[:, termdict[t]]
        y_ecfp = ecfp_labels[:, termdict[t]]

        logreg = LogisticRegression()

        auc_cnnfp = cross_val_score(logreg, cnn_fp_data, y, cv=10, scoring='roc_auc', n_jobs=-1)
        auc_ecfp = cross_val_score(logreg, ecfp_data, y_ecfp, cv=10, scoring='roc_auc', n_jobs=-1)

        writer.writerow([t, auc_cnnfp.mean(), auc_ecfp.mean()])
        print(t, auc_cnnfp.mean(), auc_ecfp.mean())
        print(datetime.datetime.now())

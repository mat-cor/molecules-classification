import datetime
import pickle
import numpy as np
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from preprocess.data_handler import load_data, categorical_labels, load_pickle
from preprocess.smiles_embedder import get_cnn_fingerprint
from preprocess.rdkutils import fp_from_smiles

path = '../data/'
termdict = load_pickle(path+'termdict.pickle')

dataset = load_data(path+'dataset.csv')
smiles = dataset['SMILES']
labels = categorical_labels(dataset['Terms'], termdict)

cnn_fp_data = get_cnn_fingerprint(smiles)
ecfp_data, valid_inds = fp_from_smiles(smiles, 2, 512, 'ecfp')
ecfp_labels = labels[valid_inds]

print(cnn_fp_data.shape, labels.shape)
print(ecfp_data.shape, ecfp_labels.shape)

with open('LRauc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Term', 'CNNFP', 'ECFP'])

    k = 0
    for i, t in enumerate(termdict.keys()):

        k += 1
        print(k)
        print(datetime.datetime.now())

        y = labels[:, i]
        y_ecfp = ecfp_labels[:, i]

        logreg = LogisticRegression()

        auc_cnnfp = cross_val_score(logreg, cnn_fp_data, y, cv=10, scoring='roc_auc', n_jobs=-1)
        auc_ecfp = cross_val_score(logreg, ecfp_data, y_ecfp, cv=10, scoring='roc_auc', n_jobs=-1)

        writer.writerow([t, auc_cnnfp, auc_ecfp])
        print(datetime.datetime.now())

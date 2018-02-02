import os
import sys
parent_path = os.path.abspath(os.path.join('..'))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import datetime
import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from preprocess.data_handler import load_data, categorical_labels, load_pickle
from preprocess.smiles_embedder import get_cnn_fingerprint
from preprocess.rdkutils import fp_from_smiles

import tensorflow as tf
from keras import backend as K

# gpu = str(sys.argv[1])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

path = '../data/'
termdict = load_pickle(path+'termdict_nopharma.pickle')

dataset = load_data(path+'dataset_nopharma.csv')
smiles = dataset['SMILES']
labels = categorical_labels(dataset['Terms'], termdict)

ecfp_data, inds = fp_from_smiles(smiles, 2, 512, 'ecfp')
cnn_fp_data = get_cnn_fingerprint(smiles[inds])
labels = labels[inds]
#
# fp_combo = [np.concatenate((c, e)) for c, e in zip(cnn_fp_data, ecfp_data)]
# fp_combo = np.array(fp_combo)

with open('../results/RFauc_nopharma.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Term', 'FP-comb'])

    k = 0
    for t in termdict.keys():

        k += 1
        print(k)
        print(datetime.datetime.now())

        y = labels[:, termdict[t]]

        rf = RandomForestClassifier(n_estimators=500)
        auc_cnn = cross_val_score(rf, cnn_fp_data, y, cv=10, scoring='roc_auc', n_jobs=-1)

        rf_e = RandomForestClassifier(n_estimators=500)
        auc_e = cross_val_score(rf, ecfp_data, y, cv=10, scoring='roc_auc', n_jobs=-1)

        writer.writerow([t, auc_cnn.mean(), auc_e.mean()])

        print(datetime.datetime.now())

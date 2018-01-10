import datetime
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from preprocess.load_data import loadDataset
from preprocess.rdkutils import smiles_list_fp

import datetime
path = '../data/'
cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'dataset.tab')

# X is a numpy matrix containing the fingerprints
# bad_smiles is a list of the SMILES that rdkit wasn't able to convert (only 2 for morgan and rdk fingerprint)
X, bad_smiles = smiles_list_fp(smiles, 2, 512, 'morgan')

#  Delete the rows related to chemicals with "bad" (=not-fingerprinted) SMILES
inds = []
for s in bad_smiles:
    inds.append(smiles.index(s))

for i in sorted(inds, reverse=True):  # Sorted in reverse order to be sure to not mess with indices
    del terms[i]


f = open(path+'LRauc_morgan2.tab', 'w')
f.write('Term\tAUCmean\n')
k = 0

# For each term, if it is present in the list of terms associated with the i-th chemical y[i] (target array) is set to 1
for term in tset:

    k += 1
    print(k)
    print(datetime.datetime.now())

    y = np.zeros(X.shape[0], dtype=int)

    for i, t_list in enumerate(terms):
        print(i)
        if term in t_list:
            y[i] = 1


    print(y.shape)
    print(y)
    # logreg = LogisticRegression()
    # auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1)

    # f.write('%s\t%5.3f\n' % (term, auc.mean()))
    # print(datetime.datetime.now())
    # print(term, auc.mean())

f.close()

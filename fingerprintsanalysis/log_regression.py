import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from chemicaldatapreprocess.load_data import loadDataset
from fingerprintsanalysis.rdkutils import smiles_list_fp

import datetime

path = '/home/mattia/Thesis/Data/'

cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'Dataset.tab')

X, bad_smiles = smiles_list_fp(smiles, 2, 1024, 'rdk')

#  Delete the rows related to chemicals with "bad" (=not-fingerprinted) SMILES
inds = []
for s in bad_smiles:
    inds.append(smiles.index(s))

for i in sorted(inds, reverse=True):  # Sorted in reverse order to be sure to not mess with indices
    del terms[i]

f = open(path+'LR_auc_rdk1024.tab', 'w')
f.write('Term\tAUCmean\tAUCstd\n')
k = 0

# For each term, if it is present in the list of terms associated with the i-th chemical y[i] (target array) is set to 1
for term in tset:

    k += 1
    print(k)
    print(datetime.datetime.now())

    y = np.zeros(X.shape[0], dtype=int)

    for t_list, i in zip(terms, range(len(terms))):
        if term in t_list:
            y[i] = 1

    logreg = LogisticRegression()
    auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1)

    f.write('%s\t%5.3f\t%5.3f\n' % (term, auc.mean(), auc.std()))
    print(datetime.datetime.now())

f.close()

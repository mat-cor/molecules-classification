import datetime
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from preprocess.rdkutils import smiles_list_fp

path = '../data/'
with open(path+'termdict.pickle', 'rb') as handle:
    termdict = pickle.load(handle)
smiles = np.load(path+'smiles.npy')
labels = np.laod(path+'multi_labels.npy')
X, bad_smiles = smiles_list_fp(smiles, 2, 512, 'rdk')

#  Delete the rows related to chemicals with "bad" (=not-fingerprinted) SMILES
inds = []
for s in bad_smiles:
    inds.append(list(smiles).index(s))

for i in sorted(inds, reverse=True):  # Sorted in reverse order to be sure to not mess with indices
    del labels[i]

f = open(path+'LR_auc_rdk512.tab', 'w')
f.write('Term\tAUCmean\n')
k = 0

# For each term, if it is present in the list of terms associated with the i-th chemical y[i] (target array) is set to 1
for i in range(labels.shape[1]):

    k += 1
    print(k)
    print(datetime.datetime.now())
    
    y = labels[:, i]
    
    logreg = LogisticRegression()
    auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=6)

    f.write('%s\t%5.3f\n' % (termdict[i], auc.mean()))
    print(datetime.datetime.now())

f.close()
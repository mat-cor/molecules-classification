import datetime
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


X = np.load('my_fp_data.npy')
y = np.load('x_seqs_labels.npy')

with open('termdict.pickle', 'rb') as handle:
    termd = pickle.load(handle)

f = open('LR_auc_my_fp.tab', 'w')
f.write('Term\tAUCmean\tAUCstd\n')
k = 0

# For each term, if it is present in the list of terms associated with the i-th chemical y[i] (target array) is set to 1
for i in range(y.shape[1]):

    k += 1
    print(k)
    print(datetime.datetime.now())

    logreg = LogisticRegression()
    auc = cross_val_score(logreg, X, y[:, i], cv=10, scoring='roc_auc')

    f.write('%s\t%5.3f\t%5.3f\n' % (termd[i], auc.mean(), auc.std()))
    print(datetime.datetime.now())

f.close()

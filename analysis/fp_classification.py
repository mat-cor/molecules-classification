import datetime
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

path = '../data/'
with open(path+'termdict.pickle', 'rb') as handle:
    termdict = pickle.load(handle)

X = np.load(path + 'smiles_rdk.npy')
labels = np.load(path + 'smiles_rdk_labels.npy')

f = open(path+'LR_auc_rdk.tab', 'w')
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

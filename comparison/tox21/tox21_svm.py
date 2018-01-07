import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold


X = np.load('my_fp_nr-ahr.npy')
y = np.load('nr-ahr_labels.npy')

clf = svm.SVC()

cv = StratifiedKFold(n_splits=10)
auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')

print('%5.3f\t%5.3f\n' % (auc.mean(), auc.std()))

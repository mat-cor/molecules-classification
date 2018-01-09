import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score

X = np.load('tetrahymena_maccs.npy')
y = np.load('tetrahymena_labels.npy')

clf = svm.SVC(kernel='rbf')

cv = StratifiedKFold(n_splits=10)
scores = cross_validate(clf, X, y, cv=cv, scoring=['roc_auc', 'accuracy', 'f1'])

print('auc: ', scores['test_roc_auc'].mean(), scores['test_roc_auc'].std())
print('acc: ', scores['test_accuracy'].mean(), scores['test_accuracy'].std())
print('f1: ', scores['test_f1'].mean(), scores['test_f1'].std())

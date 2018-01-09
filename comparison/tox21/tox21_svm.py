import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

X = np.load('nr-ahr_fp.npy')
y = np.load('nr-ahr_labels.npy')

X_test = np.load('nr-ahr_test_fp.npy')
y_test = np.load('nr-ahr_test_labels.npy')
print('Fitting...')
clf = svm.SVC(kernel='linear', C=1).fit(X, y)
print('Predicting...')
filename = 'nr-ahr-svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))

y_pred = clf.predict_proba(X_test)
auc = roc_auc_score(y_test, y_pred)
print(auc)

# cv = StratifiedKFold(n_splits=10)
# auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
# print('%5.3f\t%5.3f\n' % (auc.mean(), auc.std()))

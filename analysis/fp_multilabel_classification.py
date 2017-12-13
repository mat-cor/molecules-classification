import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from preprocess.rdkutils import smiles_list_fp

DATA_LOC = '../data/'
smiles = list(np.load(DATA_LOC+'smiles.npy'))
y = np.load(DATA_LOC+'multi_labels.npy')

X, bad_smiles = smiles_list_fp(smiles, 2, 1024, 'morgan')

inds = [smiles.index(s) for s in bad_smiles]
for i in sorted(inds, reverse=True):  # Sorted in reverse order to be sure to not mess with indices
    np.delete(y, i, 0)

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Number of train examples: ', X_train.shape[0])
print('Number of test examples: ', X_test.shape[0])
print('Multi-label classification, number of classes: ', y.shape[1])

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = np.array(clf.predict(X_test), dtype=np.int32)

print(metrics.accuracy_score(y_test, y_pred))
# print(y_test[0])
# print(y_pred[0])

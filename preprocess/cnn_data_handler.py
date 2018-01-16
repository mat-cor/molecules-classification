'''
Exctract the list of SMILES from the dataset and convert the labels to "one-hot" representation ([0 0 1 0 0 .. 0 1 0])
'''
import os
import pickle
from preprocess.data_handler import load_data, term_set
import numpy as np

import os
import sys
preprocess_path = os.path.abspath(os.path.join('..'))
if preprocess_path not in sys.path:
    sys.path.append(preprocess_path)

DATA_LOC = '../data/'
filepath = os.path.join(DATA_LOC, 'dataset_5.csv')

data = load_data(filepath)
tset = term_set(data['Terms'])
termdict = {}
for i, t in enumerate(tset):
    termdict[i] = t

labels = np.zeros((data.shape[0], len(tset)), dtype=int)

for i in range(data.shape[0]):
    for j in range(len(tset)):
        if termdict[j] in data['Terms'][i]:
            labels[i, j] = 1

np.save(DATA_LOC+'smiles_5t', data['Smiles'])
np.save(DATA_LOC+'labels_5t', labels)
with open(DATA_LOC+'termdict_5t.pickle', 'wb') as handle:
    pickle.dump(termdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

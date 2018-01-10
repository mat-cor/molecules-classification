import numpy as np
import pickle


smiles = np.load('../data/smiles.npy')
labels = np.load('../data/multi_labels.npy')

with open('../data/termdict.pickle', 'rb') as handle:
    termdict = pickle.load(handle)

j = 8237
print(smiles[j])
inds = np.where(labels[j]==1)[0]
for i in inds:
    print(termdict[i])

'''
Exctract the list of SMILES from the dataset and convert the labels to "one-hot" representation ([0 0 1 0 0 .. 0 1 0])
'''
import os
import pickle
from preprocess.load_data import loadDataset
import numpy as np

DATA_LOC = '../data/'
filepath = os.path.join(DATA_LOC, 'dataset.tab')


def getSMILES(file):
    _, smiles, _, _, _, _, _ = loadDataset(file)

    return np.array(smiles)


def get_labels(file):
    _, _, _, _, terms, _, term_set = loadDataset(file)
    term2id = {}
    id2term = {}

    for i, t in zip(range(len(term_set)), term_set):
        term2id[t] = i
        id2term[i] = t

    labels = np.zeros((len(terms), len(term_set)), dtype=int)
    for t_list, i in zip(terms, range(len(terms))):
        for term in t_list:
            j = term2id[term]
            labels[i, j] = 1

    return labels, id2term


def main():
    np.save(DATA_LOC+'smiles', getSMILES(filepath))
    labels, tdict = get_labels(filepath)
    np.save(DATA_LOC+'multi_labels', labels)
    with open(DATA_LOC+'termdict.pickle', 'wb') as handle:
        pickle.dump(tdict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

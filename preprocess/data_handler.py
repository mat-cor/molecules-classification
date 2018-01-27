import pandas as pd
import numpy as np
import pickle

def stringtolist(s):
    '''Convert a string representing a list into a list object'''
    li = []
    subs = s[1:-1].split("', '")
    for p in subs:
        li.append(p.strip("'"))
    return li


def load_data(dataset):
    data = pd.read_csv(dataset, sep='\t')
    data['Terms'] = [stringtolist(s) for s in data['Terms']]
    data['TreeIds'] = [stringtolist(s) for s in data['TreeIds']]

    return data


def get_term_set(terms):
    tl = [[t for t in tlist] for tlist in terms]
    return sorted(set(tl))


def categorical_labels(terms, tdic):
    labels = np.zeros((len(terms), len(tdic)), dtype=int)
    for i, ts in enumerate(terms):
        for t in ts:
            labels[i, tdic[t]] = 1
    return labels


def load_pickle(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

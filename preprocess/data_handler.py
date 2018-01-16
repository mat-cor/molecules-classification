import pandas as pd


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


def term_set(terms):
    tl = []
    # I want to save also a set of the terms
    for tlist in terms:
        [tl.append(t) for t in tlist]
    tset = sorted(set(tl))

    return tset

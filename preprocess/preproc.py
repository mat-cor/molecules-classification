import pickle
import csv
from preprocess.data_handler import load_data, get_term_set


def filt_terms(df, prefix):
    with open('../data/mesh_id2term.pickle', 'rb') as handle:
        id2term = pickle.load(handle)
    inds = []
    for i, tids in enumerate(df['TreeIds']):
        tids_new = list(set([t for t in tids if t.startswith(prefix)]))
        if len(tids_new):
            df.at[i, 'Terms'] = list(set([id2term[tid] for tid in tids_new]))
            df.at[i, 'TreeIds'] = tids_new
        else:
            inds.append(i)
    df = df.drop(df.index[inds])
    return df.reset_index(drop=True)


def filt_duplicates(df):
    df = df.drop_duplicates(subset='SMILES', keep='first')
    return df.reset_index(drop=True)


def filt_rares(df, freq):
    tfreq = terms_freq(df['Terms'])
    with open('../data/mesh_id2term.pickle', 'rb') as handle:
        id2term = pickle.load(handle)
    inds = []
    for i, (terms, tids) in enumerate(zip(df['Terms'], df['TreeIds'])):
        terms_new = sorted([t for t in terms if tfreq[t] >= freq])
        tids_new = [i for i in tids if id2term[i] in terms_new]
        if len(terms_new):
            df.at[i, 'Terms'] = terms_new
            df.at[i, 'TreeIds'] = tids_new
        else:
            inds.append(i)
    df = df.drop(df.index[inds])
    return df.reset_index(drop=True)


def terms_freq(terms):
    tset = get_term_set(terms)
    t2freq = {}
    for t in tset:
        c = 0
        for tl in terms:
            if t in tl:
                c += 1
        t2freq[t] = c
    return t2freq


def terms_per_cp(df):
    tpcp = {}
    for id, terms in zip(df['CID'], df['Terms']):
        tpcp[id] = len(terms)
    return tpcp


def filt_nopharma(df, prefix, tset):
    with open('../data/mesh_id2term.pickle', 'rb') as handle:
        id2term = pickle.load(handle)
    inds = []
    for i, tids in enumerate(df['TreeIds']):
        tids_new = list(set([t for t in tids if not t.startswith(prefix)]))
        if len(tids_new):
            df.at[i, 'Terms'] = list(set([id2term[tid] for tid in tids_new if id2term[tid] not in tset]))
            df.at[i, 'TreeIds'] = tids_new
        else:
            inds.append(i)
    df = df.drop(df.index[inds])
    return df.reset_index(drop=True)


if __name__ == "__main__":

    d_raw = load_data('../data/DatasetRaw.csv')
    print(d_raw.shape)
    d = filt_terms(d_raw, 'D27.505')
    d = filt_duplicates(d)
    d = filt_rares(d, 20)
    # d.to_csv('../data/dataset.csv', sep='\t', index=False)
    # fdict = terms_freq(d['Terms'])
    # with open('../data/terms_freq.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file, delimiter='\t')
    #     for key, value in fdict.items():
    #         writer.writerow([key, value])
    #
    # tset = get_term_set(d['Terms'])
    # tdict = {t: i for i, t in enumerate(tset)}
    # with open('../data/termdict.pickle', 'wb') as handle:
    #     pickle.dump(tdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    d_noph = filt_nopharma(d_raw, 'D27.505', get_term_set(d['Terms']))
    d_noph = filt_duplicates(d_noph)
    d_noph = filt_rares(d_noph, 20)
    d_noph.to_csv('../data/dataset_nopharma.csv', sep='\t', index=False)
    tset = get_term_set(d_noph['Terms'])
    tdict = {t: i for i, t in enumerate(tset)}
    with open('../data/termdict_nopharma.pickle', 'wb') as handle:
        pickle.dump(tdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
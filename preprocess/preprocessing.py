import pickle

from preprocess.load_data import loadDataset, write_dataset
from preprocess.compute_memberships import membershipMatrix, saveMatrixTab
from preprocess.frequencies import termsFrequency, saveFreq
from preprocess.exclude_compounds import *

# First load the dataset
path = '../data/'
with open(path + 'mesh_id2term.pickle', 'rb') as handle:
    id2term = pickle.load(handle)

cids_raw, smiles_raw, names_raw, formulas_raw, terms_raw, treeids_raw, tset_raw = loadDataset(path + 'DatasetRaw.tab')

# Filtering Pharmacological Action terms (D27.505)
cids, smiles, names, formulas, terms, treeids = [], [], [], [], [], []
for c, s, n, f, te, tid in zip(cids_raw, smiles_raw, names_raw, formulas_raw, terms_raw, treeids_raw):
    tid_new = [t for t in tid if t.startswith('D27.505')]
    if len(tid_new):
        cids.append(c)
        smiles.append(s)
        names.append(n)
        formulas.append(f)
        terms.append([id2term[tid] for tid in tid_new])
        treeids.append(tid_new)

# "Unique" the duplicated rows (rows with the same SMILES)
c1, s1, n1, f1, t1, tset1 = exclude_duplicate(cids, smiles, names, formulas, terms)

# Exclude terms with frequency < 20
# c, s, n, f, t, tset = exclude_rare(20, cids_u, smiles_u, names_u, formulas_u, terms_u, tset_u)
c2, s2, n2, f2, t2, tset2 = exclude_rare(20, c1, s1, n1, f1, t1, tset1)

# Save the processed data
write_dataset(path, 'dataset.tab', c2, s2, n2, f2, t2)


# Compute and save the memberships matrix
# cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'dataset.tab')
# m, term_labels = membershipMatrix(tset2, t2)
# saveMatrixTab(path+'memberships.tab', m2, term_labels2, cids)

# # Compute and save terms frequencies as a pickle dict
# f = termsFrequency(m)
# term_freq_dict = {}
#
# for i in range(len(term_labels)):
#     term_freq_dict[term_labels[i]] = f[i]
#
# with open(path+'term_freq.pickle', 'wb') as handle:
#     pickle.dump(term_freq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# saveFreq(f, term_labels, path+'termsfreq.tab')

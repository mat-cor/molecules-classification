import pickle

from preprocess.load_data import loadDataset, write_dataset
from preprocess.compute_memberships import membershipMatrix, saveMatrixTab
from preprocess.frequencies import termsFrequency, saveFreq
from preprocess.exclude_compounds import *

# First load the dataset
DATA_LOC = '../data/'

cids_raw, smiles_raw, names_raw, formulas_raw, terms_raw, treeids_raw, tset_raw = loadDataset(DATA_LOC + 'DatasetRaw.tab')

# "Unique" the duplicated rows (rows with the same SMILES)
c1, s1, n1, f1, t1, tset1 = exclude_duplicate(cids_raw, smiles_raw, names_raw, formulas_raw, terms_raw)
# Exclude the compounds with 10 < smiles_length < 400
# c1, s1, n1, f1, t1, tset1 = exclude_size(cids_u, smiles_u, names_u, formulas_u, terms_u, tset_u, 10, 400)

# Exclude terms with frequency < 20
# c, s, n, f, t, tset = exclude_rare(20, cids_u, smiles_u, names_u, formulas_u, terms_u, tset_u)
c2, s2, n2, f2, t2, tset2 = exclude_rare(20, c1, s1, n1, f1, t1, tset1)

# Save the processed data
# write_dataset(path+'dataset.tab', c2, s2, n2, f2, t2)
# write_dataset(path+'dataset30_60.tab', c2, s2, n2, f2, t2)

# Compute and save the memberships matrix
# cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'dataset.tab')
m, term_labels = membershipMatrix(tset2, t2)
# saveMatrixTab(path+'memberships.tab', m2, term_labels2, cids)

# Compute and save terms frequencies as a pickle dict
f = termsFrequency(m)
term_freq_dict = {}

for i in range(len(term_labels)):
    term_freq_dict[term_labels[i]] = f[i]

with open(DATA_LOC+'term_freq.pickle', 'wb') as handle:
    pickle.dump(term_freq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

saveFreq(f, term_labels, DATA_LOC+'termsfreq.tab')

from preprocess.load_data import loadDataset, write_dataset
from preprocess.compute_memberships import membershipMatrix, saveMatrixTab
from preprocess.frequencies import termsFrequency
from preprocess.exclude_compounds import *

# First load the dataset
path = '../data/'

cids_raw, smiles_raw, names_raw, formulas_raw, terms_raw, treeids_raw, tset_raw = loadDataset(path + 'DatasetRaw.tab')

# "Unique" the duplicated rows (rows with the same SMILES)
cids_u, smiles_u, names_u, formulas_u, terms_u = exclude_duplicate(cids_raw, smiles_raw, names_raw, formulas_raw,
                                                                   terms_raw)

tset_u = []
for t_list in terms_u:
    for t in t_list:
        tset_u.append(t)

tset_u = list(set(tset_u))  # membershipMatrix requires a list as input

# Compute the terms memberships matrix (in order to compute frequencies of the terms)
m, term_labels = membershipMatrix(tset_u, terms_u)
frequency = termsFrequency(m)

# Exclude the terms with frequency <20. If a chemical has only terms with such frequency, than it is excluded
term2freq = dict()
for fr, te in zip(frequency, term_labels):
    term2freq[te] = fr

c, s, n, f, t = exclude_rare(20, cids_u, smiles_u, names_u, formulas_u, terms_u, term2freq)

# Save the processesed dataset
# write_dataset(path+'dataset.tab', c, s, n, f, t)

# Recompute and save the memberships matrix
# cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'dataset.tab')
# m2, term_labels2 = membershipMatrix(tset, terms)
# saveMatrixTab(path+'memberships.tab', m2, term_labels2, cids)

# Exclude the compounds with 10 < smiles_length < 400

c2, s2, n2, f2, t2 = exclude_size(c, s, n, f, t, 10, 400)

# Save the processed dataset
write_dataset(path+'dataset10_400.tab', c2, s2, n2, f2, t2)


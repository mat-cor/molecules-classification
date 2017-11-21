from chemicaldatautils.frequencies import termsFrequency, termPerCompound
from chemicaldatautils.load_data import loadDataset
from chemicaldatautils.compute_memberships import membershipMatrix
import matplotlib.pyplot as plt

path = '/home/mattia/Thesis/Data/'

cids, smiles, names, formulas, terms, treeids, tset = loadDataset(path+'Dataset.tab')

m, term_labels = membershipMatrix(tset, terms)

frequency = termsFrequency(m)
terms_per_cp = termPerCompound(m)

plt.figure(1)
plt.hist(frequency, bins=30)
plt.title("Terms Frequencies Distribution")
plt.figure(2)
plt.hist(terms_per_cp, bins=7)
plt.title("Number of Terms per Compound")
plt.show()

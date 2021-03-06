import pickle
from preprocess.data_handler import load_data, term_set

path = '../data/'
with open(path + 'mesh_id2term.pickle', 'rb') as handle:
    id2term = pickle.load(handle)

data = load_data(path + 'dataset.csv')

new_tids = []
new_terms = []

for t in data['TreeIds']:
    a = list(set(['.'.join(i.split('.')[0:3]) for i in t if i.startswith('D27.505')]))
    new_tids.append(a)
    new_terms.append(list(set([id2term[j] for j in a])))

tl = []
for t in new_tids:
    [tl.append(i) for i in t]

data_new = data
data_new['Terms'] = new_terms
data_new['TreeIds'] = new_tids
data_new.to_csv(path+'dataset_5.csv', sep='\t', index=False)

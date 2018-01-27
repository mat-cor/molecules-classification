import numpy as np
import pandas as pd
import time

from preprocess.data_handler import load_data
from preprocess.rdkutils import fp_from_smiles
from analysis.molecule_embedder import get_cnn_fingerprint

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from ggplot import *
'''
t-SNE visualization comparison using CNN-FP and ECFP. 10 un-correlated terms are selected as labels
'''

# sel_terms = ['Anti-Bacterial Agents', 'Enzyme Inhibitors', 'Antineoplastic Agents', 'Antihypertensive Agents',
#              'Vasodilator Agents', 'Anti-Inflammatory Agents, Non-Steroidal', 'Anti-Arrhythmia Agents',
#              'Antiviral Agents', 'Antioxidants', 'Anticonvulsants']

sel_terms = ['Anti-Bacterial Agents', 'Enzyme Inhibitors', 'Antineoplastic Agents', 'Antihypertensive Agents',
            'Vasodilator Agents']

path = '../data/'
df = load_data(path+'dataset.csv')
sel_data = pd.DataFrame([])

for t in sel_terms:
    check = [i for i in sel_terms if i != t]
    l = []
    for i, terms in enumerate(df['Terms']):
        if t in terms and t not in check:
            if len(list(set(check)-set(terms))) == len(check):
                l.append([df.at[i, 'CID'], df.at[i, 'SMILES'], t])
    d = pd.DataFrame(l, columns=['CID', 'SMILES', 'Term'])
    sel_data = sel_data.append(d, ignore_index=True)



smiles = sel_data['SMILES']
X_ecfp, _ = fp_from_smiles(smiles, 2, 512, 'ecfp')
X = get_cnn_fingerprint(smiles)
y = sel_data['Term']

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X)
print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))
pca_50_ecfp = PCA(n_components=50)
pca_result_50_ecfp = pca_50_ecfp.fit_transform(X_ecfp)
print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50_ecfp.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
tsne_ecfp = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results_ecfp = tsne_ecfp.fit_transform(pca_result_50_ecfp)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_tsne = pd.DataFrame([])
df_tsne['x-tsne'] = tsne_results[:, 0]
df_tsne['y-tsne'] = tsne_results[:, 1]
df_tsne['label'] = y

df_tsne_ecfp = pd.DataFrame([])
df_tsne_ecfp['x-tsne'] = tsne_results_ecfp[:, 0]
df_tsne_ecfp['y-tsne'] = tsne_results_ecfp[:, 1]
df_tsne_ecfp['label'] = y

chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
        + geom_point(size=70, alpha=0.1) \
        + ggtitle("tSNE dimensions colored by term")

chart_e = ggplot(df_tsne_ecfp, aes(x='x-tsne', y='y-tsne', color='label')) \
        + geom_point(size=70, alpha=0.1) \
        + ggtitle("tSNE dimensions colored by term - ECFP")

plt.figure(1)
chart.show()
plt.figure(2)
chart_e.show()

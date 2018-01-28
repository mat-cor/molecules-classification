# molecules-classification
Small molecules classification using deep learning models

Notebooks:

- SMILES CNN Embedder: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/smiles-cnn-embedder.ipynb

- Basic Statistics: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/basic-statistics.ipynb

- CNNFP vs ECFP comparison: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/fp-comparison.ipynb

- CNNFP vs ECFP on ClinTox dataset: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/clintox-comparison.ipynb

- tSNE visualization: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/tSNE-visualization.ipynb

Packages:
- "preprocess" package contains the modules for preprocessing the data (basically discarding duplicated chemicals and non frequent terms, compute the "memberships" for the terms and convert the SMILES to fingerprints)

- "analysis" package contains fingerprint classification with LR and SMILES classification using CNN

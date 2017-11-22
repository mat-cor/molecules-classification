# molecules-classification
Small molecules classification using deep learning models


- Basic Statistics nb: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/basic-statistics.ipynb

- FP Classification nb: https://github.com/mat-cor/molecules-classification/blob/master/notebooks/fp-classification.ipynb

- LogReg AUC results: https://docs.google.com/spreadsheets/d/1cPtmbVZ20AzrsAor_i1lvfAAOZgsG_aWVfeut5zsxk0/edit?usp=sharing


Packages:
- "preprocess" package contains the modules for preprocessing the data (basically discarding duplicated chemicals and non frequent terms, compute the "memberships" for the terms and convert the SMILES to fingerprints)

- "analysis" package contains fingerprint classification with LR

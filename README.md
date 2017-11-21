# molecules-classification
Small molecules classification using deep learning models


Histograms Notebook: https://github.com/mat-cor/molecules-classification/blob/master/Histograms.ipynb
LogReg Notebook: https://github.com/mat-cor/molecules-classification/blob/master/LogisticRegression.ipynb

LogReg AUC results: https://docs.google.com/spreadsheets/d/1cPtmbVZ20AzrsAor_i1lvfAAOZgsG_aWVfeut5zsxk0/edit?usp=sharing

Packages:

"chemicaldatapreprocess" package contains the modules for preprocessing the data (basically discarding duplicated chemicals and non frequent terms, and compute the "memberships" for the terms)

"fingerprintanalysis" package contains rdk functions for converting the SMILES to fingerprints and the script for the logistic regression

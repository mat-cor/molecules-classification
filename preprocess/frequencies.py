import numpy as np


def termsFrequency(d):
    n = d.shape[0]
    f = np.empty(n, dtype=int)
    for i in range(n):
        f[i] = np.sum(d[i])
    return f


def termPerCompound(d):
    m = d.shape[1]
    freq = np.empty(m, dtype=int)
    for j in range(m):
        freq[j] = sum(d[:, j])
    return freq


def saveFreq(freq, labels, fname):
    file = open(fname, 'w')
    file.write('Terms\tFrequency\n')

    p = np.argsort(-freq)
    sorted_freq = freq[p]
    sorted_labels = labels[p]

    for label, frequency in zip(sorted_labels, sorted_freq):

        file.write(label + '\t')
        file.write(str(frequency) + '\n')

    file.close()


def saveTermPC(o, cid, fname):
    f = open(fname, 'w')
    f.write('CID\tN Terms\n')

    for i in range(o.shape[0]):
        f.write(cid[i] + '\t')
        f.write(str(o[i]) + '\n')

    f.close()

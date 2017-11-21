import numpy as np


def membershipMatrix(tset, terms):
    """ Compute a 'membership vectors' for each of the MeSH term, i.e. a vector of length = #compounds and 0/1
     values (the i-th compound is associated/not with the term)

     tset is a set of "unique" terms
     terms is a list containing lists of terms associated to each compound
    """

    matrix = np.zeros([len(tset), len(terms)], dtype=int)

    for j in range(0, len(terms)):
            for t in terms[j]:
                i = tset.index(t)
                matrix[i][j] = 1

    term_labels = np.array(tset)

    return matrix, term_labels


def saveMatrixNpy(matrix, t_labels, fname):
    np.save((fname+'_data'), matrix)
    np.save((fname+'_labels'), t_labels)


def saveMatrixTab(fname, matrix, t_labels, cids):
    f = open(fname, 'w')
    f.write('Terms')
    for id in cids:
        f.write('\t' + id)

    f.write('\n')

    for i in range(matrix.shape[0]):

        f.write(t_labels[i] + '\t')
        f.write('\t'.join(''.join(str(cell) for cell in matrix[i])))
        f.write('\n')

    f.close()


def saveTransposedMatrixTab(fname, matrix, t_labels, cids):
    f = open(fname, 'w')
    f.write('Compound')

    for lab in t_labels:
        f.write('\t' + lab)

    f.write('\n')

    for i, id in zip(range(matrix.shape[1]), cids):

        f.write(id + '\t')
        f.write('\t'.join(''.join(str(cell) for cell in matrix[:, i])))
        f.write('\n')

    f.close()


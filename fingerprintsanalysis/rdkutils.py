from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RDKFingerprint
import numpy as np


def smiles2fp(smiles, radius, fplength, t):

    ''' Error with 'C[N+](C)(C)C1=CC=CC=C1.Br[Br-]Br' '''

    m = Chem.MolFromSmiles(smiles)

    if t == 'morgan':
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=fplength)
    elif t == 'rdk':
        return RDKFingerprint(m, fpSize=fplength)
    else:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=fplength)


def fp2intarray(fp):
    fps = fp.ToBitString()

    return np.array(list(fps), dtype=int)


def smiles_list_fp(s_list, radius, fplength, t):
    '''Return a matrix representing the data and a list of the SMILES that couldn't be converted'''

    fpdata = np.empty((len(s_list), fplength), dtype=int)
    badsmiles = []

    for smile, i in zip(s_list, range(len(s_list))):
        try:
            fp = smiles2fp(smile, radius, fplength, t)
            fpdata[i] = fp2intarray(fp)
        except:
            badsmiles.append(smile)

    return fpdata, badsmiles

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RDKFingerprint
import numpy as np


def smiles2fp(smiles, radius, fplength, t):

    ''' Error with 'C[N+](C)(C)C1=CC=CC=C1.Br[Br-]Br' '''

    m = Chem.MolFromSmiles(smiles)

    if m is None:
        return None
    elif t == 'morgan' or t == 'ecfp':
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=fplength)
    elif t == 'rdk':
        return RDKFingerprint(m, fpSize=fplength)
    else:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=fplength)


def fp_from_smiles(s_list, radius, fplength, t):
    '''Return a matrix representing the data and the inds of valid mols'''
    inds = []
    fps = np.empty([len(s_list), fplength], dtype=np.int32)
    for i, smile in enumerate(s_list):
        fp = smiles2fp(smile, radius, fplength, t)
        if fp is not None:
            fpstr = fp.ToBitString()
            for j, bit in enumerate(list(fpstr)):
                fps[i, j] = bit
            inds.append(i)
    return fps[inds], inds

def exclude_rares(min_freq, cids, smiles, names, formulas, terms, t2freq):
    cids_out, smiles_out, names_out, formulas_out, terms_out = [], [], [], [], []

    for c, s, n, f, t_list in zip(cids, smiles, names, formulas, terms):
        t_list_out = []

        for t in t_list:
            if t2freq[t] >= min_freq:
                t_list_out.append(t)

        if len(t_list_out) > 0:
            cids_out.append(c)
            smiles_out.append(s)
            names_out.append(n)
            formulas_out.append(f)
            terms_out.append(t_list_out)

    return cids_out, smiles_out, names_out, formulas_out, terms_out


def exclude_duplicates(cids, smiles, names, formulas, terms):
    '''Found a set of SMILES, it will save only the first element with each SMILES'''
    smiles_set = set(smiles)
    cids_out, smiles_out, names_out, formulas_out, terms_out = [], [], [], [], []

    for sm in smiles_set:
        ind = smiles.index(sm)
        cids_out.append(cids[ind])
        smiles_out.append(smiles[ind])
        names_out.append(names[ind])
        formulas_out.append(formulas[ind])
        terms_out.append(terms[ind])

    return cids_out, smiles_out, names_out, formulas_out, terms_out

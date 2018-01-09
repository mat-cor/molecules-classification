import pickle

def stringtolist(s):
    '''Convert a string that represents a list into a list object'''
    li = []
    subs = s[1:-1].split("', '")
    for p in subs:
        li.append(p.strip("'"))
    return li


def loadOntology(ontology):
    term2id = dict()
    term2desc = dict()
    id2term = dict()

    with open(ontology, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split('\t')

            term2id[tokens[0]] = tokens[1]
            term2desc[tokens[0]] = tokens[2]

            if ';' in tokens[1]:
                treeids = tokens[1].split(';')
                for i in treeids:
                    id2term[i] = tokens[0]
            else:
                id2term[tokens[1]] = tokens[0]

    file.close()

    return [term2id, term2desc, id2term]


def loadDataset(dataset):
    cids = []
    smiles = []
    names = []
    formulas = []
    terms = []
    treeids = []
        
    with open(dataset, 'r') as file:
        next(file)
        next(file)
        next(file)
        for line in file:
            line = line.strip()
            tokens = line.split('\t')
            
            cids.append(tokens[0])
            smiles.append(tokens[1])
            names.append(tokens[2])
            formulas.append(tokens[3])
            terms.append(stringtolist(tokens[4]))
            treeids.append(stringtolist(tokens[5]))
    
    file.close()
    termslist = []
    # I want to save also a set of the terms
    for tlist in terms:
        for t in tlist:
            termslist.append(t)
    tset = list(set(termslist))
    tset.sort()

    return [cids, smiles, names, formulas, terms, treeids, tset]


def write_dataset(path, fname, cids, smiles, names, formulas, terms):
    with open(path + 'mesh_term2id.pickle', 'rb') as handle:
        term2id = pickle.load(handle)

    file = open(path+fname, 'w')
    file.write('Cid\tSmiles\tName\tFormula\tTerm_List\tTreeIds\n')
    file.write('string\tstring\tstring\tstring\tstring\tstring\n')
    file.write('\t\t\t\t\tmeta\n')

    for c, s, n, f, t_list in zip(cids, smiles, names, formulas, terms):
        tid_list = [term2id[t] for t in t_list]
        file.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (c, s, n, f, str(list(set(t_list))), str(list(set(tid_list)))))

    file.close()

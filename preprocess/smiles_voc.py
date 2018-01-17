import pickle

# openSMILES specs http://opensmiles.org/opensmiles.html
chars = ['*', 'B', 'C', 'N', 'O', 'S', 'P', 'F', 'C', 'l', 'B', 'r', 'I', 'b', 'c', 'n', 'o', 's', 'p', '[', ']', '*',
         'H', 'H', 'e', 'L', 'i', 'B', 'e', 'B', 'C', 'N', 'O', 'F', 'N', 'e', 'N', 'a', 'M', 'g', 'A', 'l', 'S', 'i',
         'P', 'S', 'C', 'l', 'A', 'r', 'K', 'C', 'a', 'S', 'c', 'T', 'i', 'V', 'C', 'r', 'M', 'n', 'F', 'e', 'C', 'o',
         'N', 'i', 'C', 'u', 'Z', 'n', 'G', 'a', 'G', 'e', 'A', 's', 'S', 'e', 'B', 'r', 'K', 'r', 'R', 'b', 'S', 'r',
         'Y', 'Z', 'r', 'N', 'b', 'M', 'o', 'T', 'c', 'R', 'u', 'R', 'h', 'P', 'd', 'A', 'g', 'C', 'd', 'I', 'n', 'S',
         'n', 'S', 'b', 'T', 'e', 'I', 'X', 'e', 'C', 's', 'B', 'a', 'H', 'f', 'T', 'a', 'W', 'R', 'e', 'O', 's', 'I',
         'r', 'P', 't', 'A', 'u', 'H', 'g', 'T', 'l', 'P', 'b', 'B', 'i', 'P', 'o', 'A', 't', 'R', 'n', 'F', 'r', 'R',
         'a', 'R', 'f', 'D', 'b', 'S', 'g', 'B', 'h', 'H', 's', 'M', 't', 'D', 's', 'R', 'g', 'C', 'n', 'F', 'l', 'L',
         'v', 'L', 'a', 'C', 'e', 'P', 'r', 'N', 'd', 'P', 'm', 'S', 'm', 'E', 'u', 'G', 'd', 'T', 'b', 'D', 'y', 'H',
         'o', 'E', 'r', 'T', 'm', 'Y', 'b', 'L', 'u', 'A', 'c', 'T', 'h', 'P', 'a', 'U', 'N', 'p', 'P', 'u', 'A', 'm',
         'C', 'm', 'B', 'k', 'C', 'f', 'E', 's', 'F', 'm', 'M', 'd', 'N', 'o', 'L', 'r', 'b', 'c', 'n', 'o', 'p', 's',
         's', 'e', 'a', 's', '@', '1', '2', '3', '-', '+', ':', '-', '=', '#', '$', ':', '/', "\\", "%", '(', ')', '.',
         '#', '(', ')', '+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'D', 'F',
         'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'W', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'g',
         'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'y']

char_set = sorted(set(chars))
print(len(char_set))
voc = {}

for i, c in enumerate(char_set):
    voc[c] = i+1

path = '../data/'
with open(path+'smiles_vocabulary.pickle', 'wb') as handle:
    pickle.dump(voc, handle, protocol=pickle.HIGHEST_PROTOCOL)

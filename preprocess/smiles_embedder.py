import pickle
import time

from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences

from preprocess.data_handler import load_pickle, load_data


def get_cnn_fingerprint(smiles):
    print('Embedding smiles...')
    start_time = time.time()
    vocabulary = load_pickle('../data/smiles_vocabulary.pickle')

    seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
    data = pad_sequences(seqs, padding='post', maxlen=1021)
    model = load_model('../analysis/fp-embedder.h5')
    embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
    fps = embedder.predict(data, batch_size=1000)
    print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps.shape[0]))
    return fps


if __name__ == "__main__":

    ds = load_data('../data/dataset.csv')
    fp = get_cnn_fingerprint(ds['SMILES'])

    fpdict = {c: f for c, f in zip(ds['CID'], fp)}

    with open('../data/cnn-fp.pickle', 'wb') as handle:
        pickle.dump(fpdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

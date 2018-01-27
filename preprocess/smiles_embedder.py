import pickle
import time

from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences

from preprocess.data_handler import load_pickle


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

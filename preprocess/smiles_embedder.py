import pickle
import time

from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences

from preprocess.data_handler import load_pickle, load_data


def get_cnn_fingerprint(smiles):
    print('Embedding smiles...')
    start_time = time.time()
    vocabulary = load_pickle('../data/smiles_vocabulary.pickle')
    start_time = time.time()
    model = load_model('../analysis/smiles-cnn-embedder.h5')
    print('Model loaded %s sec' % (time.time() - start_time))

    seqs = [[vocabulary[c] for c in list(s)] for s in smiles]
    data = pad_sequences(seqs, padding='post', maxlen=model.input_shape[1])
    embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
    fps = embedder.predict(data, batch_size=1000)
    print('Embedding complete - %s seconds, %s smiles' % (time.time() - start_time, fps.shape[0]))
    return fps

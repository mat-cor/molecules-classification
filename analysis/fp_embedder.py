import numpy as np
import sys
import os
import tensorflow as tf

from keras.models import load_model, Model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

gpu = str(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

path = '../data/'
smiles = np.load(path+'smiles.npy')

t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)
data = pad_sequences(seqs, padding='post', maxlen=1021)

model = load_model('fp-embedder.h5')
embedder = Model(inputs=model.input, outputs=model.layers[-2].output)
fps = embedder.predict(data, batch_size=1000)

np.save(path+'smiles_fp.npy', fps)
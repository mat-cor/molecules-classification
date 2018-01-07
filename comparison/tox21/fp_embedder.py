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

model = load_model('my_model.h5')

smiles = []
labels = []
fname = 'nr-ahr.smiles'
with open(fname, 'r') as file:
    for line in file:
        tks = line.split('\t')
        smiles.append(tks[0])
        labels.append(tks[2])

print(smiles[0])
print(labels[0])

t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)

data = pad_sequences(seqs, padding='post')
fp_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)
fp_output = fp_layer_model.predict(data, batch_size=1000)

print(fp_output.shape)
np.save('my_fp_nr-ahr', fp_output)
np.save('nr-ahr_labels', np.array(labels))
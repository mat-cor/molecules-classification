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
fname = 'tetrahymena.tab'
with open(fname, 'r') as file:
    next(file)
    next(file)
    next(file)
    for line in file:
        tks = line.strip().split('\t')
        smiles.append(tks[2])
        if float(tks[0]) > -0.5:
            labels.append(1)
        else:
            labels.append(0)
        
labels = np.array(labels, dtype=np.int32)

print(len(smiles[0]))
print(labels[0:5])

t = Tokenizer(filters='', lower=False, char_level=True)
t.fit_on_texts(smiles)
seqs = t.texts_to_sequences(smiles)
data = pad_sequences(seqs, padding='post', maxlen=1021)

print(len(data[0]))

fp_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)
fp_output = fp_layer_model.predict(data, batch_size=1000)

print(fp_output.shape)

np.save('tetrahymena_fp', fp_output)
np.save('tetrahymena_labels', labels)

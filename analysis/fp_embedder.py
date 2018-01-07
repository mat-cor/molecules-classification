import numpy as np
import sys
import os
import tensorflow as tf

from keras.models import load_model, Model
from keras import backend as K

gpu = str(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

model = load_model('my_model.h5')

data = np.load('x_seqs.npy')
print(data.shape)


fp_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)

fp_output = fp_layer_model.predict(data, batch_size=1000)

print(fp_output.shape)
np.save('my_fp_data', fp_output)
from keras.models import load_model

model = load_model('my_model.h5')

from keras import backend as K

get_fp_layer_output = K.function([model.layers[0].input],
                                    [model.layers[-2].output])

fp_output = get_fp_layer_output([x])[0]
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from pers_layer import *


def insert_layer(insert_id, num_filters,my_model=None, random=False):
    inp = my_model.layers[0].input
    x = my_model.layers[0].input 
    for i,layer in enumerate(my_model.layers[1:]):
        if  i == insert_id:
            sampling_size = (x.shape[1],x.shape[2])
            if num_filters == 1:
              out = Perspective_Layer(sampling_size)(x)
            else:  
              out = Perspective_Layer(sampling_size,tm=num_filters,random_init=random)(x)
              out = tf.keras.layers.Conv2D(out.shape[-1]/num_filters,1)(out) # conv 2D
            x = layer(out)
            continue
        x = layer(x)
    update_model = Model(inputs= inp,outputs= x)
    return update_model



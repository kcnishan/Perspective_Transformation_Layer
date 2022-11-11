import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *


class Perspective_Layer(tf.keras.layers.Layer): 
    def __init__(self, output_size=None,tm=1,param=8,random_init=False, **kwargs):
        self.output_size = output_size
        self.param = param
        self.tm =tm
        self.random_init = random_init
        super(Perspective_Layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_size': self.output_size,
            'param': self.param,
            'tm':self.tm,
            'random_init':self.random_init 
        })
        base_config = super(Perspective_Layer, self).get_config()
        final_config = dict(list(base_config.items()) + list(config.items()))
        return final_config
    

    def compute_output_shape(self, input_shapes):
        if type(input_shapes) is list:
            input_shapes = input_shapes[0]
        else:
            input_shapes = input_shapes
        H, W = self.output_size
        num_channels = input_shapes[-1]
        return (None, H, W, num_channels)
        

        
    def build(self, input_shape):
        if self.random_init==True:
            initial_thetas = [np.random.uniform(low=-1,high=1,size=(1, 8)) for _ in range(self.tm)]
        else:
            initial_thetas = [np.array([[1,0,0,0,1.,0,0.,0.]]) for _ in range(self.tm)]
            
            
        channel_tiles = [np.tile(initial_thetas[i],[input_shape[-1],1]).astype('float32') for i in range(self.tm)]
       
        tx = np.stack(channel_tiles, axis=0)
        
        self.wt_pers = self.add_weight(name='perspective',                                                                                                                                                            
                                  shape = (tx.shape),                                                                                                                                      
                                  initializer=tf.constant_initializer(tx),                                                                                                                              
                                  trainable=True) 
      
        super(Perspective_Layer, self).build(input_shape) 
    
        
    def call(self, inputs):
        if type(inputs) is list:
            inp = inputs[0]
        else:
            inp = inputs 
            
        expand_inp = tf.expand_dims(inp,axis=0)
        tile_inputs = tf.tile(expand_inp,multiples=[self.tm,1,1,1,1])
        
        all_out = tf.vectorized_map(self.vectorize_tms,(self.wt_pers,tile_inputs))
        all_out = tf.unstack(all_out,axis=0)
        all_out = tf.concat(all_out, axis =-1)
        return all_out
    
    
    def vectorize_tms(self,args):
        wt_tms,inps = args
        batch_size = tf.shape(inps)[0]
        channel_first_inp = K.permute_dimensions(inps, (3,1,2,0))
        singleTM_out =  tf.vectorized_map(self.vectorize_out,(wt_tms,channel_first_inp))
        singleTM_out = tf.squeeze(singleTM_out, axis=-1)
        singleTM_out = K.permute_dimensions(singleTM_out,(1,2,3,0))
        return singleTM_out

    def vectorize_out(self,arg):
        all_weights,inp = arg
        inp = K.permute_dimensions(inp, (2,0,1))
        inp2 = tf.expand_dims(inp, axis=-1)
        batch_size = tf.shape(inp2)[0]
        w_expand = tf.expand_dims(all_weights, axis=0)
        wt_tile = tf.tile(w_expand, multiples=[batch_size, 1])
        out = self.apply_transformation(inp2,wt_tile,self.output_size)
        return out

    def get_theta_matrix(self,theta):
        N = tf.shape(theta)[0]
        params_theta = tf.shape(theta)[1]
        identity_matrix = tf.eye(3)    
        identity_params = tf.reshape(identity_matrix, [3*3])
        remaining_params =  identity_params[params_theta:]
        batch_tile_remaining = tf.tile(remaining_params, [N])
        batch_params_remaining = [N, 9 - params_theta]
        batch_remaining = tf.reshape(batch_tile_remaining,batch_params_remaining )
        theta_final = tf.concat([theta, batch_remaining], axis=1)
        return theta_final
    
    
    def apply_transformation(self,features_inp, theta, out_shape=None, **kwargs):

        # get shapes of input features
        N = tf.shape(features_inp)[0]
        H = tf.shape(features_inp)[1]
        W = tf.shape(features_inp)[2]
        
        # get perspective transformation parameters
        pers_matrix = self.get_theta_matrix(theta)
        pers_matrix_shape = [N,3,3]
        pers_matrix = tf.reshape(pers_matrix, pers_matrix_shape)

        # Grid generation
        if out_shape:
            Ho = out_shape[0]
            Wo = out_shape[1]
            x_s, y_s = self.generate_grids(Ho, Wo, pers_matrix)
        else:
            x_s, y_s = self.generate_grids(H, W, pers_matrix)

        features_out = self.interpolate(features_inp, x_s, y_s)
        return features_out



    def extract_pixels(self,feature, x_vect, y_vect):

        N = tf.shape(x_vect)[0]
        H = tf.shape(x_vect)[1]
        W = tf.shape(x_vect)[2]

        batch_idx = tf.range(0, N)
        batch_idx = tf.reshape(batch_idx, (N, 1, 1))
        b_tile = tf.tile(batch_idx, (1, H, W))

        ind = tf.stack([b_tile, y_vect, x_vect], 3)

        pixels_value_out = tf.gather_nd(feature, ind)
        return pixels_value_out


    def generate_grids(self,H, W, theta):
        #get the batch size
        N = tf.shape(theta)[0] 

        # Meshgrid
        x = tf.linspace(-1.0, 1.0, W)
        y = tf.linspace(-1.0, 1.0, H)
        x_g, y_g = tf.meshgrid(x, y)

        # Flatten the meshgrid
        flatten_x_g = tf.reshape(x_g, [-1])
        flatten_y_g = tf.reshape(y_g, [-1])

        # reshape to [x_g, y_g , 1] - (homogeneous form)
        ones = tf.ones_like(flatten_x_g)
        get_grid = tf.stack([flatten_x_g, flatten_y_g, ones])

        # get grid for each features in N
        get_grid = tf.expand_dims(get_grid, axis=0)
        get_grid = tf.tile(get_grid, tf.stack([N, 1, 1]))

        theta = tf.cast(theta, 'float32')
        get_grid = tf.cast(get_grid, 'float32')

        # use matmul to transform the sampling grid
        batch_grids = tf.matmul(theta, get_grid)

        #reshape
        batch_grids = tf.reshape(batch_grids, [N, 3, H, W])

        #  homogenous coordinates to Cartesian
        omega = batch_grids[:, 2, :, :]
        x_s = batch_grids[:, 0, :, :] / omega
        y_s = batch_grids[:, 1, :, :] / omega

        return x_s, y_s

    def interpolate(self,img, x, y):
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # extract pixel value
        Ia = self.extract_pixels(img, x0, y0)
        Ib = self.extract_pixels(img, x0, y1)
        Ic = self.extract_pixels(img, x1, y0)
        Id = self.extract_pixels(img, x1, y1)


        # Finally calculate interpolated values
        # recast as float
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')

        a = (x1_f-x) * (y1_f-y)
        b = (x1_f-x) * (y1_f-y0)
        c = (x-x0_f) * (y1_f-y)
        d = (x-x0_f) * (y-y0_f)
        wa = tf.expand_dims(a, axis=3)
        wb = tf.expand_dims(b, axis=3)
        wc = tf.expand_dims(c, axis=3)
        wd = tf.expand_dims(d, axis=3)

        # compute output
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return output
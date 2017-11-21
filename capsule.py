import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils


class _Caps(base.Layer):
    """Capsule Layer.
    """
    def __init__(self, units, dim, iter_routing=2, kernel_initializer=None, trainable=True,
               name=None,**kwargs):
        super(_Caps, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.dim = dim
        self.iter_routing = iter_routing
        if kernel_initializer == None:
            self.kernel_initializer = tf.random_normal_initializer(stddev=0.01)
        else:
            self.kernel_initializer = kernel_initializer
            
    def build(self, input_shape):
        assert len(input_shape) == 3, 'Required input shape=[None, units_in, dim_in]'
        self.units_in = input_shape[1]
        self.dim_in = input_shape[2]
        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        #self.b = tf.constant(np.zeros([1, self.input_num_capsule, self.num_outputs, 1, 1]), dtype=np.float32)
        self.b = tf.get_variable('b', shape=[1, self.units_in, self.units, 1, 1], 
                                 dtype=tf.float32, initializer=tf.zeros_initializer)
        self.W = tf.get_variable('W', shape=[1, self.units_in, self.units, self.dim_in, self.dim],
                                 dtype=tf.float32, initializer=self.kernel_initializer)
        self.built = True
        
    def call(self, inputs):
        # input shape after preparation:
        #       [?, units_in, units, 1, dim_in]
        # W_tile shape: [?, units_in, units, dim_in, dim]
        inputs = tf.expand_dims(tf.expand_dims(inputs, axis=2), axis=2)
        inputs = tf.tile(inputs, [1, 1, self.units, 1, 1])
        W_tile = tf.tile(self.W, [tf.shape(inputs)[0], 1, 1, 1, 1])
        # inputs_hat: [?, units_in, units, 1, dim]
        inputs_hat = tf.matmul(inputs, W_tile)
        self.routing(inputs_hat)
        c = tf.nn.softmax(self.b, dim=2) 
        outputs = squash(tf.reduce_sum(c * inputs_hat, axis=1, keep_dims=True))
        outputs = tf.reshape(outputs, [-1, self.units, self.dim])
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = tensor_shape.TensorShape([input_shape[0], self.units, self.dim])       
        return output_shape
    
    def routing(self, inputs):
        # b shape: [1, units_in, units, 1, 1]
        # inputs:  [?, units_in, units, 1, dim]
        for i in range(self.iter_routing):
            c = tf.nn.softmax(self.b, dim=2) 
            outputs = squash(tf.reduce_sum(c * inputs, axis=1, keep_dims=True))
            self.b += tf.reduce_sum(inputs * outputs, axis=-1, keep_dims=True)
            
class _ConvCaps(base.Layer):
    """Capsule Layer.
    """
    def __init__(self, filters, dim, kernel_size, strides=(1 , 1), 
                 padding='valid', iter_routing=2, trainable=True, name=None,**kwargs):
        super(_ConvCaps, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dim = dim
        self.iter_routing = iter_routing
        
    def build(self, input_shape):
        assert len(input_shape) == 5, 'Required input shape=[None, width, height, dim, filters]'
        self.dim_in = input_shape[-2]
        self.filters_in = input_shape[-1]
        self.built = True
        
    def call(self, inputs):
        w, h = self.kernel_size
        sx, sy = self.strides
        out = tf.layers.conv3d(inputs, self.filters*self.dim, (w, h, self.dim_in), (sx, sy, 1))#, activation=tf.nn.relu)
        out = tf.reshape(out, [-1, out.shape[1].value, out.shape[2].value, self.dim, self.filters])
        return out

    def _compute_output_shape(self, input_shape):
        print("hina")
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i])
            new_space.append(new_dim)
        output_shape = tensor_shape.TensorShape([input_shape[0]] + new_space + [self.dim, self.filters])       
        return output_shape
    
    def routing(self, inputs):
        assert self.iter_routing==0, 'Routing not implemented yet'
            
def squash(tensor, axis=-1, epsilon=1e-9):
    """Squashes length of a vectors in specified input tensor's axis to the interval (0,1). 
    Arguments:
        tensor: Tensor input.
        axis: the axis to be squashed.
        epsilon: 
    Returns:
        Output tensor.
    """
    sq_norm = tf.reduce_sum(tf.square(tensor), axis, keep_dims=True)
    scale_factor = sq_norm / ((1 + sq_norm) * tf.sqrt(sq_norm + epsilon))
    out = scale_factor * tensor  
    return out


def dense(inputs, units, dim, iter_routing=2, kernel_initializer=None, trainable=True,
               name=None):
    layer = _Caps(units, dim, iter_routing=iter_routing, 
                  kernel_initializer=kernel_initializer, 
                  trainable=trainable, name=name)
    return layer.apply(inputs)


def conv2d(inputs, filters, dim, kernel_size, strides=(1 , 1), 
                 padding='valid', iter_routing=2, trainable=True, name=None):
    layer = _ConvCaps(filters, dim, kernel_size, strides=strides, iter_routing=iter_routing, 
                  trainable=trainable, name=name)
    return layer.apply(inputs)




import numpy as np
import tensorflow as tf

def margin_loss(onehot_labels, lengths, m_plus=0.9, m_minus=0.1, l=0.5):
    T = tf.to_float(onehot_labels)
    lengths = tf.to_float(lengths)
    L_present = T*tf.square(tf.nn.relu(m_plus - lengths))
    L_absent = (1-T)*tf.square(tf.nn.relu(lengths - m_minus))
    L = L_present + l*L_absent 
    return tf.losses.compute_weighted_loss(tf.reduce_sum(L, axis=1))

def reconstruction_loss(inputs, reconstruction):
    inputs_flat = tf.layers.Flatten()(inputs)
    return tf.losses.mean_squared_error(inputs_flat, reconstruction)

def mask_one(capsule_vectors, mask, is_predicting=False):
    if is_predicting:
        indices = tf.argmax(mask, axis=1)
        mask = tf.one_hot(indices=tf.cast(indices, tf.int32), depth=10)
    return tf.layers.flatten(capsule_vectors*tf.expand_dims(mask,-1), name="MaskedDigitCaps")


def decoder_nn(capsule_features, name="reconstruction"):
    if(name == None):
        name1, name2, name3 = None, None, None
    else:
        name1, name2, name3 = name+"1", name+"2", name+"3"
    fc1 = tf.layers.dense(capsule_features, 512, activation=tf.nn.relu, name=name1)
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name=name2)
    reconstruction = tf.layers.dense(fc2, 784, activation=tf.nn.sigmoid, name=name3)
    return reconstruction

def reconMashup(inputs, pred, pics_per_line=10):
    assert len(inputs) == len(pred), "need as many predictions as inputs"
    assert 2*(len(inputs))%pics_per_line == 0
    lines = int((2*len(inputs))/pics_per_line)
    h_pic = inputs[0].shape[0]
    w_pic = inputs[0].shape[1]
    h = int(h_pic*lines + lines/2)
    w = int(w_pic*pics_per_line)
    out = np.zeros((h,w),dtype=np.float32)
    startrow = 0
    endrow = h_pic 
    startcol = 0 
    endcol = pics_per_line
    i = 0
    for l in range(lines):
        if i%2 == 0:
            out[startrow:endrow] = np.hstack(inputs[startcol:endcol])
            startrow += h_pic
            endrow += h_pic+1
        else:
            out[startrow:endrow-1] = np.hstack(pred[startcol:endcol])
            startcol += pics_per_line
            endcol += pics_per_line
            out[endrow-1,:] = np.ones(w, dtype=np.float32)
            startrow += h_pic+1
            endrow += h_pic
        i+=1
    return out
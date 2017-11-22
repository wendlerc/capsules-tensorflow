import tensorflow as tf
import numpy as np
import capsule as caps
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

dataset_size = 60000
epsilon = 1e-9
regularization = True
lambda_reg = 0.0005
iter_routing = 2
num_epochs = 20
batch_size = 64
steps_per_epoch = dataset_size/batch_size
start_lr = 0.001
decay_steps = steps_per_epoch
decay_rate = 0.9
plot_num = 20
config = tf.estimator.RunConfig(save_summary_steps=100, log_step_count_steps=100)
model_dir = "/tmp/caps_mnist_sml_regularized_aug_1"

def margin_loss(onehot_labels, lengths, m_plus=0.9, m_minus=0.1, l=0.5):
    T = onehot_labels
    L_present = T*tf.square(tf.maximum(0., m_plus - lengths))
    L_absent = (1-T)*tf.square(tf.maximum(0., lengths - m_minus))
    L = L_present + l*L_absent 
    return tf.reduce_mean(tf.reduce_sum(L, axis=1), name="Margin_Loss")

def reconstruction_loss(inputs, reconstruction):
    inputs_flat = tf.layers.Flatten()(inputs)
    n = tf.cast(inputs_flat.shape[-1].value, tf.float32)
    return tf.multiply(n, tf.losses.mean_squared_error(inputs_flat, reconstruction), name="Reconstruction_Loss")

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
    

def caps_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Fashion MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # A little bit cheaper version of the capsule network in: Dynamic Routing Between Capsules
    # Std. convolutional layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=256,
        kernel_size=[9, 9],
        padding="valid",
        activation=tf.nn.relu,
        name="ReLU_Conv1")
    conv1 = tf.expand_dims(conv1, axis=-2)
    # Convolutional capsules, no routing as the dimension of the units of previous layer is one
    primarycaps = caps.conv2d(conv1, 32, 8, [9,9], strides=(2,2), name="PrimaryCaps")
    primarycaps = tf.reshape(primarycaps, [-1, primarycaps.shape[1].value*primarycaps.shape[2].value*32, 8])
    # Fully connected capsules with routing by agreement
    digitcaps = caps.dense(primarycaps, 10, 16, iter_routing=iter_routing, name="DigitCaps")
    # The length of the capsule activation vectors encodes the probability of an entity being present
    lengths = tf.sqrt(tf.reduce_sum(tf.square(digitcaps),axis=2) + epsilon, name="Lengths")
    
    # Predictions for (PREDICTION mode)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(lengths, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(lengths, name="Softmax")
    }
    
    if regularization:
        masked_digitcaps_pred = mask_one(digitcaps, lengths, is_predicting=True)
        with tf.variable_scope(tf.get_variable_scope()):
            reconstruction_pred = decoder_nn(masked_digitcaps_pred)
        predictions["reconstruction"] = reconstruction_pred
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = margin_loss(onehot_labels, lengths)
    if regularization:
        masked_digitcaps = mask_one(digitcaps, onehot_labels)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            reconstruction = decoder_nn(masked_digitcaps)
        loss += lambda_reg * reconstruction_loss(input_layer, reconstruction)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return  tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=caps_model_fn, config=config,
      model_dir=model_dir)

    # Train the model
    generator = ImageDataGenerator(width_shift_range=2/28, height_shift_range=2/28)
    train_data_im = np.reshape(train_data, [-1, 28, 28, 1])
    flow = generator.flow(train_data_im, train_labels, batch_size=num_epochs*dataset_size)
    train_data, train_labels = flow.next()
    train_data = np.reshape(train_data, [-1, 28*28])
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch*num_epochs)
  
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    accuracy_score = eval_results["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    print(eval_results)
    if regularization:
        # do some predictions and reconstructions
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data[:plot_num]},
            batch_size=batch_size,
            num_epochs=1,
            shuffle=False)
        predictions = mnist_classifier.predict(input_fn=pred_input_fn)
        
        for i, (x, p) in enumerate(zip(eval_data[:plot_num],predictions)):
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].set_title("Digit:")
            axes[0].imshow(np.reshape(x,(28,28)), cmap='gray')
            axes[1].set_title("Recon:"+str(p["classes"]))
            axes[1].imshow(np.reshape(p["reconstruction"],(28,28)),cmap='gray')
            fig.tight_layout()
            plt.show()
            
if __name__ == "__main__":
    tf.app.run()

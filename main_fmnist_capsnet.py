import tensorflow as tf
import numpy as np
import capsule as caps
from matplotlib import pyplot as plt
from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator
from load_data import load_data
from utils import reconMashup, reconstruction_loss, margin_loss, decoder_nn, mask_one

dataset_size = 60000
epsilon = 1e-9
regularization = True
lambda_reg = 0.4 # ~28*28*0.0005
iter_routing = 2 # routing 2 in this implementation corresponds to routing 3 in the paper
num_epochs = 100
batch_size = 128
steps_per_epoch = dataset_size/batch_size
steps_train = steps_per_epoch*num_epochs 
start_lr = 0.001
decay_steps = steps_per_epoch
decay_rate = 0.9
plot_num = 100
config = tf.estimator.RunConfig(save_summary_steps=100, log_step_count_steps=100)
model_dir = "/tmp/fmnist/r2_reg1"
mapfn_parallel_iterations = batch_size


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
    digitcaps = caps.dense(primarycaps, 10, 16, iter_routing=iter_routing, mapfn_parallel_iterations=mapfn_parallel_iterations, name="DigitCaps")
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
    tf.summary.scalar("margin_loss", loss)    
    if regularization:
        masked_digitcaps = mask_one(digitcaps, onehot_labels)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            reconstruction = decoder_nn(masked_digitcaps)
        rec_loss = reconstruction_loss(input_layer, reconstruction)
        tf.summary.scalar("reconstruction_loss", rec_loss)    
        loss += lambda_reg * rec_loss
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Summary hook
        summary_hook = tf.train.SummarySaverHook(
                save_steps=config.save_summary_steps,
                output_dir=model_dir,
                summary_op=tf.summary.merge_all())
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return  tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])
        
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    (x_train, y_train), (x_test, y_test) = load_data()
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
              'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    img_rows, img_cols = x_train[0].shape
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255    
    
    x_train = np.reshape(x_train, (-1, img_rows*img_cols))
    x_test = np.reshape(x_test, (-1, img_rows*img_cols))
    
    # Load training and eval data
    train_data = x_train  # Returns np.array
    train_labels = y_train
    eval_data = x_test  # Returns np.array
    eval_labels = y_test

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=caps_model_fn, config=config,
      model_dir=model_dir)

    # Train the model #
    # Data Augumentation
    generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
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
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_train)
  
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    if regularization:
        # do some predictions and reconstructions
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data[:plot_num]},
            batch_size=batch_size,
            num_epochs=1,
            shuffle=False)
        predictions = mnist_classifier.predict(input_fn=pred_input_fn)
        prediction_pics = [np.reshape(p['reconstruction'], (28,28)) for p in predictions]
        eval_pics = np.reshape(eval_data[:plot_num], (-1, 28, 28))
        plt.imshow(reconMashup(eval_pics, prediction_pics), cmap='gray')
        plt.axis('off')
            
if __name__ == "__main__":
    tf.app.run()

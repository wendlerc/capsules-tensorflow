import argparse
import tensorflow as tf
import numpy as np
import capsule as caps
from matplotlib import pyplot as plt
from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator
from utils import reconMashup, reconstruction_loss, margin_loss, decoder_nn, mask_one

dataset_size = 60000
epsilon = 1e-12
lambda_reg = 0.4 # ~ 0.0005*28*28
iter_routing = 2 # routing 2 in this implementation corresponds to routing 3 in the paper
learn_coupling = False # decides whether the initial coupling logits are learned or not
num_epochs = 5#100
batch_size = 128
steps_per_epoch = dataset_size/batch_size
steps_train = steps_per_epoch*num_epochs 
start_lr = 0.001
decay_steps = steps_per_epoch
decay_rate = 0.95
plot_num = 100
config = tf.estimator.RunConfig(save_summary_steps=100, log_step_count_steps=100)
model_dir = None#"/tmp/mnist/r2_reg04_bfix_stopgrad"
mapfn_parallel_iterations = 16#batch_size
    

def caps_model_fn(features, labels, mode):
    hooks = []
    train_log_dict = {}
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
    digitcaps = caps.dense(primarycaps, 10, 16, iter_routing=iter_routing, learn_coupling=learn_coupling, mapfn_parallel_iterations=mapfn_parallel_iterations, name="DigitCaps")
    # The length of the capsule activation vectors encodes the probability of an entity being present
    lengths = tf.sqrt(tf.reduce_sum(tf.square(digitcaps),axis=2) + epsilon, name="Lengths")
    
    # Predictions for (PREDICTION mode)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(lengths, axis=1),
        "probabilities": tf.nn.softmax(lengths, name="Softmax")
    }

    if lambda_reg > 0:
        masked_digitcaps_pred = mask_one(digitcaps, lengths, is_predicting=True)
        with tf.variable_scope(tf.get_variable_scope()):
            reconstruction_pred = decoder_nn(masked_digitcaps_pred)
        predictions["reconstruction"] = reconstruction_pred
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    m_loss = margin_loss(onehot_labels, lengths)
    train_log_dict["margin loss"] = m_loss
    tf.summary.scalar("margin_loss", m_loss)        
    if lambda_reg > 0:
        masked_digitcaps = mask_one(digitcaps, onehot_labels)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            reconstruction = decoder_nn(masked_digitcaps)
        rec_loss = reconstruction_loss(input_layer, reconstruction)
        train_log_dict["reconstruction loss"] = rec_loss
        tf.summary.scalar("reconstruction_loss", rec_loss)   
        loss = m_loss + lambda_reg * rec_loss
    else:
        loss = m_loss
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Logging hook
        train_log_dict["accuracy"] = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])[1]
        logging_hook = tf.train.LoggingTensorHook(train_log_dict, 
                                                  every_n_iter=config.save_summary_steps)
        # Summary hook
        summary_hook = tf.train.SummarySaverHook(
                save_steps=config.save_summary_steps,
                output_dir=model_dir,
                summary_op=tf.summary.merge_all())
        hooks += [logging_hook, summary_hook]
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return  tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=hooks)
        
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
    #set some global variables
    dataset_size = len(train_data)
    steps_per_epoch = dataset_size/batch_size
    steps_train = steps_per_epoch*num_epochs
    decay_steps = steps_per_epoch
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=caps_model_fn, config=config,
      model_dir=model_dir)

    # Train the model #
    # Data Augumentation
    if num_epochs > 0:
        generator = ImageDataGenerator(width_shift_range=0.071, height_shift_range=0.071)
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
    if lambda_reg > 0 and plot_num > 0:
        # do some predictions and reconstructions
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data[:plot_num]},
            batch_size=batch_size,
            num_epochs=1,
            shuffle=False)
        predictions = mnist_classifier.predict(input_fn=pred_input_fn)
        prediction_pics = [np.reshape(p['reconstruction'], (28,28)) for p in predictions]
        eval_pics = np.reshape(eval_data[:plot_num], (-1, 28, 28))
        mash = reconMashup(eval_pics, prediction_pics)
        plt.imshow(mash, cmap='gray')
        plt.axis('off')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    parser.add_argument("--lambda_reg", type=float, default=0.4, help="Weight of the regularization term (reconstruction loss).")
    parser.add_argument("--iter_routing", type=int, default=3, help="Amount of routing iterations between PrimaryCaps and DigitCaps.")
    parser.add_argument("--learn_coupling", action="store_true", help="Flag that determines whether to learn the initial coupling logits or not.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Amount epochs used for training (0 corresponds to no training).")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size used for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The initial learning rate for ADAM.")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="The learning rate decay rate (1 corresponds to no decay).")
    parser.add_argument("--plot_num", type=int, default=100, help="The amount of example pictures to be plotted (only relevant when lambda_reg > 0).")
    parser.add_argument("--model_dir", type=str, default=None, help="The directory in which the model parameters and tensorboard logs are to be stored.")
    parser.add_argument("--mapfn_parallel_iterations", type=int, default=128, help="The amount of parallel iterations for the tf.mapfn call in DigitCaps. If this parameter is set to zero tf.tile and tf.matmul is used instead.")

    args = parser.parse_args()
    lambda_reg = args.lambda_reg 
    iter_routing = max(args.iter_routing - 1, 0) 
    learn_coupling = args.learn_coupling
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    start_lr = args.learning_rate
    decay_rate = args.decay_rate
    plot_num = args.plot_num
    model_dir = args.model_dir
    mapfn_parallel_iterations = args.mapfn_parallel_iterations
    tf.app.run()

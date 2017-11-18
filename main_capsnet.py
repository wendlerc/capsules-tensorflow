import tensorflow as tf
import numpy as np
import capsule as caps
from load_data import load_data

epsilon = 1e-9
  
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
        activation=tf.nn.relu)
    conv1 = tf.expand_dims(conv1, axis=-2)
    # Convolutional capsules, no routing as the dimension of the units of previous layer is one
    primarycaps = caps.conv2d(conv1, 32, 8, [9,9], strides=(2,2))
    primarycaps = tf.reshape(primarycaps, [-1, primarycaps.shape[1].value*primarycaps.shape[2].value*32, 8])
    # Fully connected capsules with routing by agreement
    digitcaps = caps.dense(primarycaps,10,16)
    # The length of the capsule activation vectors encodes the probability of an entity being present
    logits = tf.sqrt(tf.reduce_sum(tf.square(digitcaps),
                              axis=2, keep_dims=True) + epsilon)
    logits = tf.squeeze(logits)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        estimator = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        estimator =  tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
    # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        estimator = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return estimator


def main(unused_argv):
    (x_train, y_train), (x_test, y_test) = load_data()
    
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
              'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    img_rows, img_cols = x_train[0].shape
    
    ''' Normalizing data '''
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255    
    
    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)
    
    
    # Load training and eval data
    train_data = x_train  # Returns np.array
    train_labels = y_train
    eval_data = x_test  # Returns np.array
    eval_labels = y_test
    
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=caps_model_fn, 
      model_dir="/tmp/caps_new")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=10,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn)
  
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    accuracy_score = mnist_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

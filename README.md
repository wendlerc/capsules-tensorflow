# capsules-tensorflow
Another implementation of Hinton's capsule networks in tensorflow.

At the moment the implementation provides the means to set up the model presented in https://arxiv.org/abs/1710.09829.

## Implemented (capsule.py): 
* fully connected capsule layers with routing
* convolutional capsule layers without routing

## TODO:
* convoutional capsule layers with routing
* add decoder net for regularization
* speed things up

For a minimal example please have a look at: main_mnist_capsnet.py. In this file a CapsNet with the architecture proposed in the
linked paper is trained on the MNIST dataset (without regularization). Alternatively, if you have keras installed you can train the same CapsNet on the fashion MNIST dataset by running main_capsnet.py.

I am new to tensorflow, therefore, feedback regarding coding style or mistakes is very appreciated!

# capsules-tensorflow
Another implementation of Hinton's capsule networks in tensorflow. At the moment the implementation provides the means to set up the model presented in [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829). Everything seems to work now. Accuracy with 3 routing iterations, regularization, a batch size of 256 and 20 epochs of training: 99,5%. Lager experiments are performed at the moment.

## Preliminary results:
Here are some reconstructions of testset digits obtained by the encoder of a capsnet trained on MNIST for 20 epochs with regularization and iter_routing=2:

![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/results_20epochs.png "odd lines: inputs, even lines: reconstructions"| width=560))


The odd lines contain the input digits and the even lines the reconstructed digits. For the reconstruction the orientation of the longest DigitCapsule vector is used.

## Implemented: 
* fully connected capsule layers with routing (capsule.py)
* convolutional capsule layers without routing (capsule.py)
* regularization (main_mnist_capsnet.py)

## TODO:
* train for many epochs
* convoutional capsule layers with routing
* speed things up

For a minimal example please have a look at: main_mnist_capsnet.py. In this file a CapsNet with the architecture proposed in the
linked paper is trained on the MNIST dataset. 


I am new to tensorflow, therefore, feedback regarding coding style or mistakes is appreciated!

### Other Implementations:
* Clean Keras implementation: [Xifeng Guo](https://github.com/XifengGuo/CapsNet-Keras)

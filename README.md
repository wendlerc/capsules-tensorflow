# capsules-tensorflow
Another implementation of Hinton's capsule networks in tensorflow.

At the moment the implementation provides the means to set up the model presented in [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Preliminary results:
Here are some reconstructions of testset digits obtained by the encoder of a capsnet trained on MNIST for 2 epochs with regularization and iter_routing=2:

![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_0.png "0")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_1.png "1")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_2.png "2")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_3.png "3")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_4.png "4")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_5.png "5")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_6.png "6")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_7.png "7")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_8.png "8")
![alt text](https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/regularization1_routing2_epochs2/recon_9.png "9")


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

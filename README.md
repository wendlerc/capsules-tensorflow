# capsules-tensorflow
Another implementation of Hinton's capsule networks in tensorflow. At the moment the implementation provides the means to set up the model presented in [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829). Everything seems to work now. Accuracy with 3 routing iterations, regularization, a batch size of 256 and 20 epochs of training: 99,5%. Larger experiments are performed at the moment.

## Preliminary results:
Here are some reconstructions of testset digits obtained by the encoder of a capsnet trained on MNIST for 20 and 100 epochs with regularization and iter_routing=2:

<img src="https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/results_20epochs.png" width="280" heigth="570"> <img src="https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/results_100epochs.png" width="280" heigth="570"> 


The odd lines contain the input digits and the even lines the reconstructed digits. For the reconstruction the orientation of the longest DigitCapsule vector is used.

Reconstructions for the fashion MNIST dataset after 10 epochs with regularization and iter_routing=2:

<img src="https://github.com/chrislybaer/capsules-tensorflow/blob/master/results/results_10epochs_fmnist.png" width="280" heigth="570"> 

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

<!--
  Title: Capsule Networks 
  Description: A tensorflow implementation of Hinton's capsule networks.
  -->

<meta name='keywords' content='capsules, hinton, capsnet, capsules tensorflow, capsnet tensorflow'>

Copyright 2017 Chri Wendler

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

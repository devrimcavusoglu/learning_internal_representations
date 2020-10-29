# Backpropagation Walkthrough

The content of this project is to walk one through the paper which is one of the classics among the deep learning papers, [Learning Internal Representations by Error Propagation](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/readings/Learning%20representations%20by%20back-propagating%20errors.pdf) in 1986 by [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart), [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) and [Ronald Williams](https://en.wikipedia.org/wiki/Ronald_J._Williams). The paper had shown that the perceptrons are worth working on as they introduced *Generalized Delta Rule* which is a generalized version of the *Standard Delta Rule* to be used in deep networks, and these networks are capable of learning internal representations. They also provided simulation results for the problems which are earlier faced, and led [Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky) and [Papert](https://en.wikipedia.org/wiki/Seymour_Papert) to be pessimistic about the perceptrons. 

The research in deep learning has boomed since the introduction of the back-propagation algorithm. This paper brought lives to many newly created neural network architectures that the world currently uses. The paper also has a discussion on Recurrent Nets at the end of the paper that is worth reading.

# Contents

Here, we have prepared a set of content which we think are useful for understanding the methods introduced in the paper. The repo have three main components,

* [Walkthrough of the paper](./learning_internal_representations.ipynb)
* [Neural Net Models for Example Datasets](./example_datasets.ipynb)
* [Neural Network Code](./nnet.py)

In the walkthrough notebook you will find a notebook designed in a way to ease the equations or methods for the reader. Also, we preserved the original notation used in the paper, so that the reader can easily keep up with the paper. However, we will include a dictionary for common use of terms.

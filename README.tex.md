
# Multilayer Perceptron (MLP)

A multilayer perceptron (MLP) is a class of feed-forward artificial neural network(NN). A MLP consists of, at least, three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function(Wikipedia).
In this repository, I present the mathematical formulation and implementation in Python of a MLP. I also train e validate the algorithm against three different data sets, presenting pratical examples of how to use MLP to classify data.

# Preamble 

You may be asking yourself: why do we need another MLP explanation in the internet? This repository  provides my thought process after reading several materials when I tried to implement a MLP by myself. At the time, I could understand and implement it only after a lot of reading, and trial and error. So, as I felt the necessity to read different points of views and be exposed to different ways of explaining the same topic, I think others may face the same situation.

Hope this document can help you on your learning journey. Good Luck !

# Mathematical Formulation

MLPs are composed by mathematical neurons and its synapses, in this case called weights. Neurons are arranged in layers, and connected between them through weights. The simplest MLP you can build is composed of three layers: Input, Hidden and Output layers. In the classical topology each neuron of a given layer is fully connected with the neurons of the next layer. 

Our ultimate goal is to mathematically formulate a MLP, however there is a simple type of neural network that will help you to build the foundation to understand MLPs. *Perceptron* is single neuron NN as shown in picture bellow.  

The picture bellow shows a *Perceptron* and its different mathematical components:

 <p align="center"> 
    <img src="doc/perceptron.png" alt="Perceptron">
 </p>

Mathematically speaking, this neuron produces the following output:

\begin{equation*}
    $\out(t) = tau( $\sum_{i=1}^{n} w_{i}*x_{i}$ + b )$
\end{equation*}

In other words, the output of a neuron is given by a linear combination of its inputs:

\begin{equation*}
$\sum_{i=1}^{n} w_{i}*x_{i} :(1)$
\end{equation*}

Adjusted by an offset, called baias, which give us output **a**:

2.
\begin{equation*}
$a = \sum_{i=1}^{n} w_{i}*x_{i} + b :(2)$
\end{equation*}

Then, the output is calculated passing the input to a function denominated **Activation Function**:

\begin{equation*}
$z = out(t) = \tau(a) :(3)$
\end{equation*}

If you remind of Linear Algebra, the equation *(2)* looks very similar to a hyperplane. Moreover, the equation 
give us a notion of how far the data sample *X\<x1,x2,x3,...,xn\>* is from the hyperplane:

\begin{equation*}
$\sum_{i=1}^{n} w_{i}*x_{i} + b = 0 :(4)$
\end{equation*}

Using *Percepron*, we can create a classifier that given an example characterized by the input *X<x1,x2,x3,...,xn>*, it returns if the example is **Class** **A = 0** or **B = 1**, using as decisive factor how far the point is from the hyperplane. If you noticed, this is the role of the **Activation Function** in the equation *(3)*. In this case, the example shows the step function, but as I'll show you later there are better **Activation Functions** that we can use.

## Now, you should be wondering: How does perceptron "learn" the best hyperplane? 

Indeed, the challenge in Machine Learning is: how do we "learn"? *Perceptron* classifiers is a *supervised learning algorithm*, therefore we must provide a set or examples beforehand, from which we'll calculate the best possible hyperplane that separates the examples into two different classes. As you noticed, a single neuron is capable of classifying only two classes. Another characteristic of *Perceptron* is that, it works well only with linearly separable datasets.

Two sets of points are said to be linear separable if there is at least one hyperplane that can separate them in two classes. In two dimensional spaces, you can think as a line that can separate the points on a plane on two different sides. You can read more in [Linear separability - Wikepedia.](https://en.wikipedia.org/wiki/Linear_separability)


## Stochastic Gradient Descent (SGD) - How NNs Learn

Neural Networks, including *Perceptron* and *MLP*, apply the method *Stochastic Gradient Descent (SGD)*  on their learning process. SGD is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization You can find a more formal explanation in [Wikepedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

It may sound confusing, even intimidating. But don't worry we'll get there.

Simplifying, SGD is an algorithm to estimate the minimum of a function.
SGD is used a lot on the optimization field. In optimization, the ultimate goal is to minimize the *Cost function*, which is the measure of how far we are from the goal. If you have ever studied optimization problems that sounds familiar, if you never heard of optimization problems that is not a problem at all.

The concept of *Cost Function* is also applicable to NNs, and it's mathematically express how far we are from the ultimate goal. The ultimate goal in Classification problems is to define how far we are of classifying the examples correctly.

Let's make a hypothetical experiment. Let's say we have a data set with 10 examples, given by: 

\begin{equation*}
$Xi = <x1, x2, x3, ...., Xn, Y> :(5)$
\end{equation*}

where, *<x1, x2, x3, ...., Xn>* is the input and *Y* is the correct class for the example. Now, we randomly generates a set of initial weights <w1, w2, w3, ..., wn> and biases <b1, b2, b3,..., bn>. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. On vary popular **Cost Function** is the quadratic error difference, given by:

\begin{equation*}
$C(w, b) = \|\|Y -Ŷ\|\|^2 :(4)$
\end{equation*}

This formula tells that, for a given set of wights and biases (w,b), the cost is the distance between the right classification *Y* and the estimated classification *Ŷ* squared. On 1-dimensional problems, such as *Perceptron*, the distance is simply the difference, on N-dimensional problems the value is the module of the vectorial distance between the two vectors.

In this context, SGD is a method to update *(w,b)* interactively towards one of the minimum of the function *C(w,b)*. SGD defines the following two update equations, also called in this article learning equations:

\begin{equation*}
$w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(6)$
\end{equation*}

\begin{equation*}
$b_i(t+1) = b_i(t) - \eta\frac{\partial C}{\partial b_i} :(7)$
\end{equation*}

These two equations tells that we must every interaction of the algorithm we update the weights and biases by a fraction *$\eta$*

# Topology 

We'll start formulating a MLP with the following topology: 2-2-1
* 2 Input Layer Neurons
* 2 Hidden Layer neurons
* 1 Output Layer Neuron

Then we'll generalize this particular case to have a general formulation for a general topology.

 <p align="center"> 
    <img src="doc/mlp-topology.png" alt="MLP Topology">
 </p>



# Implementation

# Training and Validating

# Example MLP Library usage

## XOR Gate

## Iris UCI

## MNNIST
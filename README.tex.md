# Multilayer Perceptron (MLP)

A multilayer perceptron (MLP) is a class of feed-forward artificial neural network(NN). A MLP consists of, at least, three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function (Wikipedia).

In this repository, I present the mathematical formulation and implementation in Python of a MLP. I also train and validate the algorithm against three different data sets, presenting practical examples of how to use MLP to classify data.

# Preamble 

You may be asking yourself: why do we need another Perceptron/MLP explanation in the internet? This repository provides my thought process after reading several materials when I tried to implement a MLP myself. At the time, I was able to understand and implement it only after a lot of reading, and trial and error. So, as I felt the necessity to be exposed to different ways of explaining the same topic, I think others may face the same situation.

Hope this document can help you on your learning journey. Good Luck !

# Mathematical Formulation

MLPs are composed by mathematical neurons and its synapses, in this case called weights. Neurons are arranged in layers, and connected between them through weights. The simplest MLP you can build is composed of three layers: Input, Hidden and Output layers. In the classical topology each neuron of a given layer is fully connected with the neurons of the next layer. 

## *Perceptron*

Many of the concepts utilized in this articles are explained in the [Perceptron](https://github.com/filipecalasans/percepetron) repository. So, you may want to check it out before continuing to the MLP formulation. Perceptron is the simplest Neural Network composed of a single neuron that helps us to build the theoretical foundation for MLPs. However, if you already have solid understanding of the mathematic concepts used on *Perceptrons*, feel free to skip to the next section.

# Topology 

We'll start formulating a MLP with the topology 2-2-1 as shown in the picture billow, then we'll generalize from this particular case.

The topology is composed of:

* 2 Input Layer Neurons
* 2 Hidden Layer neurons
* 1 Output Layer Neuron 

 <p align="center"> 
    <img src="doc/mlp-topology.png" alt="MLP Topology">
 </p>

# Formulation

## Notation

We are going to use the following notation across this article:

* $w_{ij}^L$: Weight connection between the Neuron number $j$ of the layer $L-1$ (previous layer) and Neuron number $i$ of the layer $L$ (current layer).
* $b_{i}^L$ neuron bias $i$ of the layer $L$ 
* $x_{i}$: Component $i$ of the input vector.
* $y_{i}$: Component $i$ of the expected output vector.
* $ŷ_{i}$: Component $i$ of the estimated output vector.
* $z_{i}^L$: Output of the Neuron $i$ in the layer $L$ before applying the activation function.
* $a_{i}^L$:Output of the neuron $i$ in the layer $L$ after applying the activation function.
* $\sigma$: activation function
## Matrix Notation

* $W^L$: Weight connection matrix of Layer the $L$.
* $B^L$: bias vector of the layer $L$. 
* $X$: Input vector
* $Y$: Expected output vector. This vector represents a known class.
* $Ŷ$: Estimated output vector. This vector represents the computed output of the Neural Network.
* $Z^L$: Neuron Output vector before applying the activation function.
* $A^L$: Neuron Output vector after applying the activation function.
* $T$: Training Example: The tuple $T=<X,Y>$ defines a training example.

Let's write the equations for the particular case: 2-2-1 MLP.

### Hidden Layer

$$
z_{1}^h = w_{11}^hx_{1} + w_{12}^hx_{2} + b_{1}^h
$$

$$
z_{2}^h = w_{21}^hx_{1} + w_{22}^hx_{2} + b_{2}^h
$$

$$
a_{1}^h = \sigma(z_{1}^h) 
$$

$$
a_{2}^h = \sigma(z_{2}^h) 
$$

Matrix notation:

$$
\begin{bmatrix}
z_{1}^h \\
z_{2}^h 
\end{bmatrix}
=
\begin{bmatrix}
w_{11}^h && w_{12}^h \\
w_{21}^h && w_{22}^h  
\end{bmatrix}
\begin{bmatrix}
x_{1}^h \\
x_{2}^h 
\end{bmatrix}
+
\begin{bmatrix}
b_{1}^h \\
b_{2}^h 
\end{bmatrix}
$$


$$
\begin{bmatrix}
a_{1}^h \\
a_{2}^h 
\end{bmatrix}
=
\begin{bmatrix}
\sigma(z_{1}^h) \\
\sigma(z_{2}^h) 
\end{bmatrix}
=
\sigma(
\begin{bmatrix}
z_{1}^h \\
z_{2}^h 
\end{bmatrix}
)
$$

Algebric matrix equation:

$$
Z^L = W^LX+B^L
$$


$$
A^L = \sigma(Z^L)
$$

### Output Layer

The same formulation can be applied for the output layer. However,the input will be the previous layer's output $A^{L-1}$.

$$
Z^O = W^OA^{L-1}+B^O
$$

$$
A^O = \sigma(Z^O)
$$

### Generalized Notation

We can generalize the previous formulas to any neural network topology. We can assume that the weight matrix for the input layer is identity matrix $I$, and the bias matrix is zero. Then, we can use a single equation to represent the output of a given layer L.

$$
Z^L = W^LA^{L-1}+B^L
$$

$$
A^L = \sigma(Z^L)
$$

$$
A^L = \sigma(W^LA^{L-1}+B^L)
$$


# Example MLP Library usage



## XOR Gate

## Iris UCI

## MNNIST
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
* $T$: Training Example; The tuple $T=<X,Y>$ defines a training example.

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

We can generalize the previous formulas to any neural network topology. We can assume that the weight matrix for the input layer is the identity matrix $I$, and the bias matrix is zero. Then, we can use a single equation to represent the output of a given layer L.

$$
Z^L = W^LA^{L-1}+B^L
$$

$$
A^L = \sigma(Z^L)
$$

$$
A^L = \sigma(W^LA^{L-1}+B^L)
$$

# Backpropagation

Backpropagation is the mechanism used to update the weights and bias starting from the output, and propagating through the other layers.

Let's start applying the Stochastic Gradient Descent in the output layer.

$$
W^o(t+1) = W^o(t) - \eta\frac{\partial C}{\partial W^o}
$$

$$
B^o(t+1) = B^o(t) - \eta\frac{\partial C}{\partial B^o}
$$

where,

$$
C = \|Y-Ŷ\|^2
$$

Applying the Chain Rule in the derivative, we have:

$$
\frac{\partial C}{\partial W^o} = \frac{\partial C}{\partial A^L} \frac{\partial A^L}{\partial W^o}
$$

$$
\frac{\partial C}{\partial B^o} = \frac{\partial C}{\partial A^L} \frac{\partial A^L}{\partial B^o}
$$

Now, you should remember that $A^L = \sigma(Z^L)$, therefore we can apply the chain rule one more time. Then, we have:

$$
\frac{\partial C}{\partial W^o} = \frac{\partial C}{\partial A^o} \frac{\partial A^o}{\partial W^o} = \frac{\partial C}{\partial A^o} \frac{\partial A^o}{\partial Z^o} \frac{\partial Z^o}{\partial W^o}  
$$

$$
\frac{\partial C}{\partial B^o} = \frac{\partial C}{\partial A^o} \frac{\partial A^o}{\partial B^o} = \frac{\partial C}{\partial A^o} \frac{\partial A^o}{\partial Z^o} \frac{\partial Z^o}{\partial B^o}  
$$

If you remember from vectorial calculus, you can notice that:

$$
\frac{\partial C}{\partial A^o} = \frac{\partial C(Y,Ŷ)}{\partial Ŷ} =\nabla{C}
$$

This is true because the cost function is scalar and the derivative of the Cost Function regarding each component $Ŷ$ is by definition the gradient of $C$.

Moreover, we can draw the following simplifications:

$$
A^L = \sigma(Z^L) \rightarrow\frac{\partial A^L}{\partial Z^L} = \sigma'(Z^L )
$$

$$
Z^L = W^LA^{L-1} + B^L
$$

$$
z_i^L=\sum_{j=1}^{n} w_{ij}^L*a^{L-1}_{j}+b^L_i\rightarrow\frac{\partial z_i^L}{\partial w_{ij}^L}=\frac{\partial}{\partial w_{ij}^L}(\sum_{j=1}^{n} w_{ij}^L*a^{L-1}_{j}+b^L_i)=a^{L-1}_j 
$$

$$
z_i^L=\sum_{j=1}^{n} w_{ij}^L*a^{L-1}_{j}+b^L_i\rightarrow\frac{\partial z_i^L}{\partial b_{i}^L}=\frac{\partial}{\partial b_{i}^L}(\sum_{j=1}^{n} w_{ij}^L*a^{L-1}_{j}+b^L_i)=1
$$

Using algebric matrix notation:

$$
\frac{\partial Z^L}{\partial W^L}=A^{L-1} 
$$

$$
\frac{\partial Z^L}{\partial B^L}=1
$$

Applying the generic formulas above on the output layer, we have:

$$
\frac{\partial Z^o}{\partial W^o}=A^{h} 
$$

$$
\frac{\partial Z^o}{\partial B^o}=1
$$

#### *Hadamard Product*

Before we merge the equations into the learning equation (SGD), let me introduce you the *Hadamard Product* operator if you already are not familiar with it. So, we can present the eqaution on a more compact way.

The *Hadamard Product*, symbol $\circ$, is an operation between two matrices of same dimension, that produces a matrix with the same dimension. The result matrix is the result of the multiplication of elements $i,j$ of the original matrices. Therefore, it is a element-wise multiplication between two matrices. For example:

$$
\begin{bmatrix}
a_{11}&a_{12}\\
a_{21}&a_{22} \\
a_{31}&a_{32} 
\end{bmatrix}
\circ
\begin{bmatrix}
b_{11}&b_{12}\\
b_{21}&b_{22} \\
b_{31}&b_{32} 
\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11}&a_{12}b_{12}\\
a_{21}b_{21}&a_{21}b_{22} \\
a_{31}b_{31}&a_{32}b_{32} 
\end{bmatrix}
$$

#### Continuing with the mathematical formulation...

$$
\frac{\partial C}{\partial W^o}=\nabla C\circ\sigma'(Z^o)A^h
$$

$$
\frac{\partial C}{\partial B^o}=\nabla C\circ\sigma'(Z^o)
$$

You might have noticed that the Chain Rule allowed us to write the derivatives above as a function of two terms:

* The first term depends only on the output layer: $\nabla C\circ\sigma(Z^o)$.
* The second term depends only on the previous layer output: $A^h$

In other words, the chain rules enabled us to backpropagate the error, the first term, to the previous layer. We then can introduce a new term called delta:

$$
\delta^o=\nabla C\circ\sigma(Z^o)
$$

Therefore,

$$
\frac{\partial C}{\partial W^o}=\delta^oA^h
$$

$$
\frac{\partial C}{\partial B^o}=\delta^o
$$

Then, we have the following learning equations for the output layer:

$$
W^o(t+1)=W^o(t)-\eta\delta^oA^{h}
$$

$$
B^o(t+1)=B^o(t)-\eta\delta^o
$$


### Generalized Learning Equations

We are now ready to generalize the equations for any neural network topology. 

Starting from the derivatives, we have:

$$
\frac{\partial C}{\partial W^L}=\frac{\partial C}{\partial A^L} \frac{\partial A^L}{\partial Z^L}\frac{\partial Z^L}{\partial W^L}
$$

$$
\frac{\partial C}{\partial B^L}=\frac{\partial C}{\partial A^L} \frac{\partial A^L}{\partial Z^L}\frac{\partial Z^L}{\partial B^L}
$$

The key is to understand that we can calculate $\frac{\partial C}{\partial A^L}$ easily only when $L$ is the output layer. Intuitively, you may be asking yourself: what if we could be able to write that derivative as function of $\frac{\partial C}{\partial A^o}$ which we know how to calculate?

Alright, in fact that is the mechanism that characterize the backpropagation algorithm. We'll leverage Chain Rule one more time to expand the derivatives.

$$
\frac{\partial C}{\partial W^L}=\frac{\partial C}{\partial A^o} \frac{\partial A^L}{\partial Z^{L}}\frac{\partial Z^L}{\partial W^L}
$$

$$
\frac{\partial C}{\partial B^L}=\frac{\partial C}{\partial A^L} \frac{\partial A^L}{\partial Z^L}\frac{\partial Z^L}{\partial B^L}
$$


$$
\frac{\partial C}{\partial W^L}=\delta^LA^{L-1}
$$

$$
\frac{\partial C}{\partial B^L}=\delta^L
$$

Finally, we have the generalized learning equations for any layer:

$$
W^L(t+1)=W^L(t)-\eta\delta^LA^{L-1}
$$

$$
B^L(t+1)=B^L(t)-\eta\delta^L
$$

Once again, the Chain Rule is fundamental to understand MLPs. It provides us the mathematical tool implement the MLP backpropagation algorithm. Shortly, we can think that we are going to calculate the estimates output and the update the weights and biases depending on the error status. These two steps will be calculated until we consider the network trained.

# Example MLP Library usage

## XOR Gate

## Iris UCI

## MNNIST
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

* <img src="/tex/e7aef05c93fc141752370e7884d53cf7.svg?invert_in_darkmode&sanitize=true" align=middle width=22.523917349999987pt height=27.6567522pt/>: Weight connection between the Neuron number <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> of the layer <img src="/tex/abf17eec3c78fcd0a21c1803f1ad3c5a.svg?invert_in_darkmode&sanitize=true" align=middle width=39.49764389999999pt height=22.465723500000017pt/> (previous layer) and Neuron number <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of the layer <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> (current layer).
* <img src="/tex/3cf887b76cd63f28a0b450e37d0b0957.svg?invert_in_darkmode&sanitize=true" align=middle width=16.073120249999988pt height=27.6567522pt/> neuron bias <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of the layer <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> 
* <img src="/tex/dc80c8df8d6a3120a158fb62653b1321.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>: Component <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of the input vector.
* <img src="/tex/e46f5aa3f3f039ebf21a80fd0cf8fad9.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/>: Component <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of the expected output vector.
* <img src="/tex/80c16818f5bacc9e6c2df624c82478fd.svg?invert_in_darkmode&sanitize=true" align=middle width=13.30009889999999pt height=22.831056599999986pt/>: Component <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of the estimated output vector.
* <img src="/tex/ce778e6bc5924581f2331d9858245900.svg?invert_in_darkmode&sanitize=true" align=middle width=17.38594274999999pt height=27.6567522pt/>: Output of the Neuron <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> in the layer <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> before applying the activation function.
* <img src="/tex/09be2af89e4dda1eb9b38aaa3e5d9a24.svg?invert_in_darkmode&sanitize=true" align=middle width=17.70747824999999pt height=27.6567522pt/>:Output of the neuron <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> in the layer <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> after applying the activation function.
* <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/>: activation function
## Matrix Notation

* <img src="/tex/57b2cba25e280b2de8bf26312cb12268.svg?invert_in_darkmode&sanitize=true" align=middle width=26.826581099999988pt height=27.6567522pt/>: Weight connection matrix of Layer the <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/>.
* <img src="/tex/094f53458217273445ccce7baf12d6ac.svg?invert_in_darkmode&sanitize=true" align=middle width=22.31172734999999pt height=27.6567522pt/>: bias vector of the layer <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/>. 
* <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>: Input vector
* <img src="/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/>: Expected output vector. This vector represents a known class.
* <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/>: Estimated output vector. This vector represents the computed output of the Neural Network.
* <img src="/tex/fe79bd004eef79327b7ab06a50349f2a.svg?invert_in_darkmode&sanitize=true" align=middle width=21.41559254999999pt height=27.6567522pt/>: Neuron Output vector before applying the activation function.
* <img src="/tex/8bb3167ecf0fa0108755856809faee3d.svg?invert_in_darkmode&sanitize=true" align=middle width=21.34712249999999pt height=27.6567522pt/>: Neuron Output vector after applying the activation function.
* <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/>: Training Example; The tuple <img src="/tex/461d196408c16404dbc26594468bd95d.svg?invert_in_darkmode&sanitize=true" align=middle width=98.44159214999999pt height=22.465723500000017pt/> defines a training example.

Let's write the equations for the particular case: 2-2-1 MLP.

### Hidden Layer

<p align="center"><img src="/tex/d7708558c35c81d779cd31487977a226.svg?invert_in_darkmode&sanitize=true" align=middle width=178.66631145pt height=18.84197535pt/></p>

<p align="center"><img src="/tex/6703fbd3aeae884740fa45bcbe4a67be.svg?invert_in_darkmode&sanitize=true" align=middle width=178.66631145pt height=18.84197535pt/></p>

<p align="center"><img src="/tex/988505df6489ecbbaa8d3bf22a458665.svg?invert_in_darkmode&sanitize=true" align=middle width=78.77860155pt height=18.88772655pt/></p>

<p align="center"><img src="/tex/a374c448d293c60a4003639b733ee15f.svg?invert_in_darkmode&sanitize=true" align=middle width=78.77860155pt height=18.88772655pt/></p>

Matrix notation:

<p align="center"><img src="/tex/5f792519e82c030da96be50cae5e466a.svg?invert_in_darkmode&sanitize=true" align=middle width=248.79407355pt height=39.60032339999999pt/></p>


<p align="center"><img src="/tex/73ee4131875a3113112c3b149fbf7cbe.svg?invert_in_darkmode&sanitize=true" align=middle width=192.4050843pt height=39.60032339999999pt/></p>

Algebric matrix equation:

<p align="center"><img src="/tex/b06fed030a14f77906a3dd28687bdbf2.svg?invert_in_darkmode&sanitize=true" align=middle width=129.11521919999998pt height=16.0201668pt/></p>


<p align="center"><img src="/tex/b476e3c976491c89cd024121a9741db6.svg?invert_in_darkmode&sanitize=true" align=middle width=89.0924892pt height=18.7598829pt/></p>

### Output Layer

The same formulation can be applied for the output layer. However,the input will be the previous layer's output <img src="/tex/12d0b3f47becd314d9dc6b1d1e206cc4.svg?invert_in_darkmode&sanitize=true" align=middle width=38.173690499999985pt height=27.6567522pt/>.

<p align="center"><img src="/tex/f4440a33c20e911a9276102e5f8e8274.svg?invert_in_darkmode&sanitize=true" align=middle width=157.20465585pt height=16.0201668pt/></p>

<p align="center"><img src="/tex/85cf3dbb171380e7420424d3eb189e62.svg?invert_in_darkmode&sanitize=true" align=middle width=91.7608131pt height=18.7598829pt/></p>

### Generalized Notation

We can generalize the previous formulas to any neural network topology. We can assume that the weight matrix for the input layer is the identity matrix <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/>, and the bias matrix is zero. Then, we can use a single equation to represent the output of a given layer L.

<p align="center"><img src="/tex/7b262a0b10ae8325622072f36a36c9ab.svg?invert_in_darkmode&sanitize=true" align=middle width=153.20212874999999pt height=16.0201668pt/></p>

<p align="center"><img src="/tex/b476e3c976491c89cd024121a9741db6.svg?invert_in_darkmode&sanitize=true" align=middle width=89.0924892pt height=18.7598829pt/></p>

<p align="center"><img src="/tex/f03ef7c446880adcf27bf1653b5b428b.svg?invert_in_darkmode&sanitize=true" align=middle width=176.72388854999997pt height=18.7598829pt/></p>

# Backpropagation

Backpropagation is the mechanism used to update the weights and bias starting from the output, and propagating through the other layers.

Let's start applying the Stochastic Gradient Descent in the output layer.

<p align="center"><img src="/tex/256310140947dc655eeb18b50c3e842d.svg?invert_in_darkmode&sanitize=true" align=middle width=203.48346809999998pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/9723cd2559f18d936c61399e74ea908f.svg?invert_in_darkmode&sanitize=true" align=middle width=189.9389052pt height=33.81208709999999pt/></p>

where,

<p align="center"><img src="/tex/05e93be16a148556571a1f27e18d4cd1.svg?invert_in_darkmode&sanitize=true" align=middle width=104.31716624999999pt height=19.68035685pt/></p>

Applying the Chain Rule in the derivative, we have:

<p align="center"><img src="/tex/2caf4772ebf495a752e13bf1bfcb3c74.svg?invert_in_darkmode&sanitize=true" align=middle width=131.13572835pt height=36.22493325pt/></p>

<p align="center"><img src="/tex/34f5801dced2c1e21dff63852f08724f.svg?invert_in_darkmode&sanitize=true" align=middle width=123.67117124999999pt height=36.22493325pt/></p>

Now, you should remember that <img src="/tex/02dd8e3137727ab0e389ea816026ece8.svg?invert_in_darkmode&sanitize=true" align=middle width=89.09248919999999pt height=27.6567522pt/>, therefore we can apply the chain rule one more time. Then, we have:

<p align="center"><img src="/tex/be0509beecf95b53fd44ddd6043964b7.svg?invert_in_darkmode&sanitize=true" align=middle width=255.746205pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/dd10555169dc8c2116845317b625af06.svg?invert_in_darkmode&sanitize=true" align=middle width=242.2016421pt height=33.81208709999999pt/></p>

If you remember from vectorial calculus, you can notice that:

<p align="center"><img src="/tex/671775388633277a95f4bb8804fb560b.svg?invert_in_darkmode&sanitize=true" align=middle width=171.96540075pt height=40.4538486pt/></p>

This is true because the cost function is scalar and the derivative of the Cost Function regarding each component <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/> is by definition the gradient of <img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>.

Moreover, we can draw the following simplifications:

<p align="center"><img src="/tex/98b663b87645be54368b95958a8a39c0.svg?invert_in_darkmode&sanitize=true" align=middle width=222.02152335pt height=36.22493325pt/></p>

<p align="center"><img src="/tex/2c1cee6e6437dfca834777a0ead7ab8b.svg?invert_in_darkmode&sanitize=true" align=middle width=153.20212874999999pt height=16.0201668pt/></p>

<p align="center"><img src="/tex/4987ca62df17e64167820e4b356c37c9.svg?invert_in_darkmode&sanitize=true" align=middle width=506.1073974pt height=47.1348339pt/></p>

<p align="center"><img src="/tex/60c2eddb37c068e5a8bf0b82d604f0d6.svg?invert_in_darkmode&sanitize=true" align=middle width=468.20389109999996pt height=47.1348339pt/></p>

Using algebric matrix notation:

<p align="center"><img src="/tex/a3fcd323b1d2b2191b4347b586d53bab.svg?invert_in_darkmode&sanitize=true" align=middle width=99.35279474999999pt height=36.22493325pt/></p>

<p align="center"><img src="/tex/9e6689f8ef9cc870d37e91a82cad0d73.svg?invert_in_darkmode&sanitize=true" align=middle width=64.88345985pt height=36.22493325pt/></p>

Applying the generic formulas above on the output layer, we have:

<p align="center"><img src="/tex/7fbf35c81eb1d516cc2484eafd647cc0.svg?invert_in_darkmode&sanitize=true" align=middle width=78.67418955pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/1f58b6f4fc0a92330bc6ed76852c9038.svg?invert_in_darkmode&sanitize=true" align=middle width=62.3537211pt height=33.81208709999999pt/></p>

#### *Hadamard Product*

Before we merge the equations into the learning equation (SGD), let me introduce you the *Hadamard Product* operator if you already are not familiar with it. So, we can present the eqaution on a more compact way.

The *Hadamard Product*, symbol <img src="/tex/c0463eeb4772bfde779c20d52901d01b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/>, is an operation between two matrices of same dimension, that produces a matrix with the same dimension. The result matrix is the result of the multiplication of elements <img src="/tex/4fe48dde86ac2d37419f0b35d57ac460.svg?invert_in_darkmode&sanitize=true" align=middle width=20.679527549999985pt height=21.68300969999999pt/> of the original matrices. Therefore, it is a element-wise multiplication between two matrices. For example:

<p align="center"><img src="/tex/24f1cd2e07cf122f11be49de772765db.svg?invert_in_darkmode&sanitize=true" align=middle width=326.90295pt height=59.1786591pt/></p>

#### Continuing with the mathematical formulation...

<p align="center"><img src="/tex/594122240c165d40f0eb5d464666229d.svg?invert_in_darkmode&sanitize=true" align=middle width=167.91042345pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/bba598cb7af5c491427b0b8e8e0c8904.svg?invert_in_darkmode&sanitize=true" align=middle width=143.37074565pt height=33.81208709999999pt/></p>

You might have noticed that the Chain Rule allowed us to write the derivatives above as a function of two terms:

* The first term depends only on the output layer: <img src="/tex/d29683749fccfb0bd54b46e2962014d7.svg?invert_in_darkmode&sanitize=true" align=middle width=84.62435849999999pt height=24.65753399999998pt/>.
* The second term depends only on the previous layer output: <img src="/tex/2aaf50857a2f748cdb2ec38a570daded.svg?invert_in_darkmode&sanitize=true" align=middle width=20.02482569999999pt height=27.91243950000002pt/>

In other words, the chain rules enabled us to backpropagate the error, the first term, to the previous layer. We then can introduce a new term called delta:

<p align="center"><img src="/tex/a4aa6ca14c8b21155b7c3f2e40c247e3.svg?invert_in_darkmode&sanitize=true" align=middle width=121.78056pt height=16.438356pt/></p>

Therefore,

<p align="center"><img src="/tex/0837f1cc7a6f812b6595d7d1230b5e43.svg?invert_in_darkmode&sanitize=true" align=middle width=93.91275959999999pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/ce9ea307d5c7651e59ea2e021be2e138.svg?invert_in_darkmode&sanitize=true" align=middle width=68.55118545pt height=33.81208709999999pt/></p>

Then, we have the following learning equations for the output layer:

<p align="center"><img src="/tex/412121646fd9418d0965574e6116b0f7.svg?invert_in_darkmode&sanitize=true" align=middle width=202.01512815pt height=18.88772655pt/></p>

<p align="center"><img src="/tex/07716b37e364b0824a5001b92261821b.svg?invert_in_darkmode&sanitize=true" align=middle width=172.13869859999997pt height=16.438356pt/></p>


### Generalized Learning Equations

We are now ready to generalize the equations for any neural network topology. 

Starting from the derivatives, we have:

<p align="center"><img src="/tex/ddd4f883afde2e21e3fa8585812dcbbc.svg?invert_in_darkmode&sanitize=true" align=middle width=172.01831295pt height=36.22493325pt/></p>

<p align="center"><img src="/tex/ab539152ea96a13fafe2f05ecd6319d2.svg?invert_in_darkmode&sanitize=true" align=middle width=162.98860545pt height=36.22493325pt/></p>

The key is to understand that we can calculate <img src="/tex/834af103ce7a0ff92a35477f70949bcb.svg?invert_in_darkmode&sanitize=true" align=middle width=26.289506099999997pt height=28.92634470000001pt/> easily only when <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> is the output layer. Intuitively, you may be asking yourself: what if we could be able to write that derivative as function of <img src="/tex/1c223dc72ead1e5363ade9018afc0c4b.svg?invert_in_darkmode&sanitize=true" align=middle width=24.211835999999995pt height=28.92634470000001pt/> which we know how to calculate?

Alright, in fact that is the mechanism that characterize the backpropagation algorithm. We'll leverage Chain Rule one more time to expand the derivatives.

<p align="center"><img src="/tex/ee2c7d645f0717f6da3e70ec9127d16b.svg?invert_in_darkmode&sanitize=true" align=middle width=169.4885742pt height=36.22493325pt/></p>

<p align="center"><img src="/tex/ab539152ea96a13fafe2f05ecd6319d2.svg?invert_in_darkmode&sanitize=true" align=middle width=162.98860545pt height=36.22493325pt/></p>


<p align="center"><img src="/tex/6fb74a3d7fe8ed8d07c63378c5335f5b.svg?invert_in_darkmode&sanitize=true" align=middle width=117.1211052pt height=33.81208709999999pt/></p>

<p align="center"><img src="/tex/f79a0b8a2bb723941dc6b642a514a185.svg?invert_in_darkmode&sanitize=true" align=middle width=73.61064644999999pt height=33.81208709999999pt/></p>

Finally, we have the generalized learning equations for any layer:

<p align="center"><img src="/tex/f2a98a44175bab6b1e95763c93362885.svg?invert_in_darkmode&sanitize=true" align=middle width=227.75321415pt height=18.7598829pt/></p>

<p align="center"><img src="/tex/e952f2e45aa0662a9d94d8171e8419ed.svg?invert_in_darkmode&sanitize=true" align=middle width=179.7279pt height=18.7598829pt/></p>

Once again, the Chain Rule is fundamental to understand MLPs. It provides us the mathematical tool implement the MLP backpropagation algorithm. Shortly, we can think that we are going to calculate the estimates output and the update the weights and biases depending on the error status. These two steps will be calculated until we consider the network trained.

# Example MLP Library usage

## XOR Gate

## Iris UCI

## MNNIST
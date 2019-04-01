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
* <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/>: Training Example: The tuple <img src="/tex/461d196408c16404dbc26594468bd95d.svg?invert_in_darkmode&sanitize=true" align=middle width=98.44159214999999pt height=22.465723500000017pt/> defines a training example.

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

We can generalize the previous formulas to any neural network topology. We can assume that the weight matrix for the input layer is identity matrix <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/>, and the bias matrix is zero. Then, we can use a single equation to represent the output of a given layer L.

<p align="center"><img src="/tex/7b262a0b10ae8325622072f36a36c9ab.svg?invert_in_darkmode&sanitize=true" align=middle width=153.20212874999999pt height=16.0201668pt/></p>

<p align="center"><img src="/tex/b476e3c976491c89cd024121a9741db6.svg?invert_in_darkmode&sanitize=true" align=middle width=89.0924892pt height=18.7598829pt/></p>

<p align="center"><img src="/tex/f03ef7c446880adcf27bf1653b5b428b.svg?invert_in_darkmode&sanitize=true" align=middle width=176.72388854999997pt height=18.7598829pt/></p>


# Example MLP Library usage



## XOR Gate

## Iris UCI

## MNNIST
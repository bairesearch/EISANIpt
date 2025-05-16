# EISANIpt/EIANNpt

### Author

Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

### Description

Excitatory Inhibitory Sequentially/Summation Activated Neuronal Input (EISANI) and Excitatory Inhibitory neuron Artificial Neural Network (EIANN) for PyTorch - experimental

#### EISANI Algorithm

##### Comparison with SANI

The EISANI algorithm differs from the original SANI (sequentially activated neuronal input) specification in two ways, but is equivalent otherwise (dynamic network generation etc);

1. tabular/image datasets use summation activated neuronal input. A sequentially activated neuronal input requirement is not enforced, as this was designed for sequential data such as NLP (text).
2. both excitatory and inhibitory input are used (either !useEIneurons:excitatory/inihibitory synapses or useEIneurons:excitatory/inhibitory neurons). 

##### Advantages

EISANI algorithm advantages/biologically feasibility over classical ANN (artificial neural network);

- no backpropagation.
- supports continuous learning.
- training speed (can learn meaningful representations/discrimination even from a single example if overtrained).
- neural substrate can be reapplied to any dataset/task (only need to train new connections to target output neurons).
- supports independent excitatory/inhibitory neurons.
- low power requirements (binary processing could be embedded in hardware/ASIC architecture).
- online learning (unbatched, single epoch).

##### Summary

1. useDynamicGeneratedHiddenConnections=True: initialise a sparsely connected neural network as having no hidden layer neurons or connections (input layer and output layer class target neurons only). useDynamicGeneratedHiddenConnections=False: randomly initialise a sparsely connected neural network with input layer, hidden layer and output layer class target neurons.
2. For each dataset sample propagate the input through layers of the sparsely connected multilayer network. 
3. useDynamicGeneratedHiddenConnections=True: generate hidden neurons (segments) within the sparsely connected multilayer network, where the weights of the generated hidden neurons (segments) represent a subset of the previous layer distribution (activations).
4. the network has binary excitatory and inhibitory weights or neurons.
5. connect the activated hidden neurons to class target output neurons. Output connections can be weighted based on the number of times a hidden neuron is activated for a given class target.
6. prediction is performed based on the most activated output class target neurons. 
7. can subsequently prune the network to retain a) the most prevalent (highest weighted) and predictive (most exclusive) output connections and b) their respective hidden neurons.

See [EISANIpt_EISANImodel.nlc](https://github.com/bairesearch/EIANNpt/blob/master/EISANIpt/EISANIpt_EISANImodel.nlc?raw=true) for detailed natural language code (specification).

##### Future

- test image dataset learning (reuse lower layers as convolutional kernels for different parts of an image).
- test sequential/NLP dataset learning (and compare non-sequential summation with original sequentially activated neuronal input SANI implementation).

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision
pip install torchsummary
pip install networkx
pip install matplotlib
```

### Execution
```
source activate pytorchsenv
python ANNpt_main.py
```

## EISANI vs Backprop performance

![EISANIbackpropTestAccuracy-SMALL.png](https://github.com/bairesearch/EIANNpt/blob/master/graph/EISANIbackpropTestAccuracy-SMALL.png?raw=true)

Tests conducted with default settings;
```
useDynamicGeneratedHiddenConnections = True
useDynamicGeneratedHiddenConnectionsVectorised = True
useDynamicGeneratedHiddenConnectionsUniquenessChecks = False
useEIneurons = False
useSparseMatrix = True
numberOfSynapsesPerSegment = 5
useGrayCode = True
continuousVarEncodingNumBits = 8
continuousVarMin = 0.0
continuousVarMax = 1.0
segmentActivationThreshold = numberOfSynapsesPerSegment-2
targetActivationSparsityFraction = 0.1
useBinaryOutputConnections = True
useActiveBias = True
hiddenLayerSizeSANI = 1280000
numberNeuronsGeneratedPerSample = 5
```

To increase train/test performance, increase;
```
continuousVarEncodingNumBits
hiddenLayerSizeSANI
numberNeuronsGeneratedPerSample
```

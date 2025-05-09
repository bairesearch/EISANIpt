# EISANIpt/EIANNpt

### Author

Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

### Description

Excitatory/Inhibitory neuron summation activated neuronal input/artificial neural network (EISANI/EIANN) for PyTorch

#### EISANI Algorithm

The EISANI algorithm differs from the original SANI (sequentially activated neuronal input) specification in two ways. The EISANI algorithm is equivalent to the original SANI specification otherwise (dynamic network generation etc);

1. tabular/image datasets use summation activated neuronal input. A sequentially activated neuronal input requirement is not enforced, as this was designed for sequential data such as NLP (text).
2. both excitatory and inhibitory input are used (either !useEIneurons:excitatory/inihibitory synapses or useEIneurons:excitatory/inhibitory neurons). 

EISANI algorithm advantages/biologically feasibility over classical ANN (artificial neural network);

- no backpropagation.
- supports continuous learning.
- training speed (can learn meaningful representations/discrimination even from a single example if overtrained).
- neural substrate can be reapplied to any dataset/task (only need to train new connections to target output neurons).
- supports independent excitatory/inhibitory neurons.
- low power requirements (binary processing could be embedded in FPGA like architecture).
- online learning (unbatched).

Future:
- test image dataset learning (reuse lower layers as convolutional kernels for different parts of an image).
- test sequential/NLP dataset learning (and compare summation with original sequentially activated neuronal input SANI implementation).

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
useDynamicGeneratedHiddenConnectionsUniquenessChecks = True
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

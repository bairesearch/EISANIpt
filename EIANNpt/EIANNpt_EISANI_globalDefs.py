"""EIANNpt_EISANI_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EIANNpt EISANI globalDefs

"""

debugEISANIoutput = False

useDynamicGeneratedHiddenConnections = True	#dynamically generate hidden neuron connections (else use randomly initialised hidden connections)
if(useDynamicGeneratedHiddenConnections):
	useDynamicGeneratedHiddenConnectionsVectorised = True	#execute entire batch simultaneously
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = True
useEIneurons = False	#use separate excitatory and inhibitory neurons (else use excitatory and inhibitory connections/synapses)
useSparseMatrix = True	#use sparse tensors to store connections (else use dense tensors)
numberOfSynapsesPerSegment = 5	#default: 5	#exp: 15	#number of input connections per neuron "segment"; there is 1 segment per neuron in this implementation
useGrayCode = True	#use graycode to encode continuous vars into binary (else use thermometer encoding)
continuousVarEncodingNumBits = 8	#default: 8	#number of bits to encode a continuous variable to
continuousVarMin = 0.0	#sync with datasetNormaliseMinMax
continuousVarMax = 1.0	#sync with datasetNormaliseMinMax
segmentActivationThreshold = numberOfSynapsesPerSegment-2	#default: numberOfSynapsesPerSegment-2 (ie 3; allowing for 1 mismatch redundancy), or numberOfSynapsesPerSegment (ie 5; allowing for 0 mismatch redundancy)	#minimum net activation required for neuron to fire (>= value), should be less than numberOfSynapsesPerSegment	#total neuron z activation expected from summation of excitatory connections to previous layer neurons
	
targetActivationSparsityFraction = 0.1	#ideal number of neurons simultaneously active per layer
useBinaryOutputConnections = True	#use binary weighted connections from hidden neurons to output neurons
useActiveBias = True	#bias positive (ceil) for odd k
hiddenLayerSizeSANI = 1280000	#heuristic: >> hiddenLayerSizeTypical * continuousVarEncodingNumBits
numberNeuronsGeneratedPerSample = 5	#default: 5	#heuristic: hiddenLayerSizeSANI//numberOfSynapsesPerSegment  	#for higher train performance numberNeuronsGeneratedPerSample should be increased substantially (eg 50), however this assigns a proportional number of additional neurons to the network (limited by hiddenLayerSizeSANI)

trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/EIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEISANI'


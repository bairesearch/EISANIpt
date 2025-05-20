"""EISANIpt_EISANI_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt globalDefs

"""

debugEISANIoutput = False
debugEISANIfastTrain = False
debugMeasureClassExclusiveNeuronRatio = True	#measure ratio of a) class (output neuron) exclusive hidden neurons to b) non class (output neuron) exclusive hidden neurons
debugMeasureRatioOfHiddenNeuronsWithOutputConnections = True	#measure ratio of hidden neurons with output connections to those without output connections

useDefaultNumNeuronsParam = True	#default: True (use low network width)
useDefaultSegmentSizeParam = True	#default: True (use moderate segment size/num synapses)
useDefaultNumLayersParam = True	#default: True (use low num layers)
useInitOrigParam = False	#use original test parameters

useImageDataset = False
if(useImageDataset):
	CNNkernelSize = 3
	CNNstride = 1
	CNNkernelThreshold = 5 #(ie sum of applied kernel is >= 5)
	CNNmaxPool = True
	EISANICNNinputChannelThreshold = 0.5
	EICNNoptimisationSparseConv = False	#default: True
	EICNNoptimisationBlockwiseConv = False	#default: False (old optimisation)
	EICNNoptimisationPackBinary = False	#default: False (old optimisation)
	EICNNoptimisationAssumeInt8 = False	#cnn operations are not currently implemented on CuDNN
	if(EICNNoptimisationBlockwiseConv or EICNNoptimisationPackBinary):
		EICNNoptimisationAssumeInt8 = True	#int8 is only used for specific CNN intermediate optimisations
else:
	useTabularDataset = True
	
useDynamicGeneratedHiddenConnections = True	#dynamically generate hidden neuron connections (else use randomly initialised hidden connections)
if(useDynamicGeneratedHiddenConnections):
	useDynamicGeneratedHiddenConnectionsVectorised = True	#execute entire batch simultaneously
useEIneurons = False	#use separate excitatory and inhibitory neurons (else use excitatory and inhibitory connections/synapses)
useSparseMatrix = True	#use sparse tensors to store connections (else use dense tensors)
useGrayCode = True	#use graycode to encode continuous vars into binary (else use thermometer encoding)
continuousVarMin = 0.0	#sync with datasetNormaliseMinMax
continuousVarMax = 1.0	#sync with datasetNormaliseMinMax

targetActivationSparsityFraction = 0.1	#ideal number of neurons simultaneously active per layer
useBinaryOutputConnections = True	#use binary weighted connections from hidden neurons to output neurons
if(useDefaultNumNeuronsParam):
	continuousVarEncodingNumBits = 8	#default: 8	#number of bits to encode a continuous variable to	#for higher train performance numberNeuronsGeneratedPerSample should be increased (eg 16), however this requires a high numberNeuronsGeneratedPerSample+hiddenLayerSizeSANI to capture the larger number of input variations
	hiddenLayerSizeSANI = 1280000	#heuristic: >> hiddenLayerSizeTypical * continuousVarEncodingNumBits
	numberNeuronsGeneratedPerSample = 5	#50	#default: 5	#heuristic: hiddenLayerSizeSANI//numberOfSynapsesPerSegment  	#for higher train performance numberNeuronsGeneratedPerSample should be increased substantially (eg 50), however this assigns a proportional number of additional neurons to the network (limited by hiddenLayerSizeSANI)
else:
	continuousVarEncodingNumBits = 16
	hiddenLayerSizeSANI = 1280000*2
	numberNeuronsGeneratedPerSample = 50

if(useDefaultSegmentSizeParam):
	numberOfSynapsesPerSegment = 5	#default: 5	#exp: 15	#number of input connections per neuron "segment"; there is 1 segment per neuron in this implementation
	segmentActivationThreshold = 3	#default: 3; allowing for 1 inhibited mismatch redundancy or 2 non inhibited mismatch redundancy	#minimum net activation required for neuron to fire (>= value), should be less than numberOfSynapsesPerSegment	#total neuron z activation expected from summation of excitatory connections to previous layer neurons
	useActiveBias = True	#bias positive (ceil) for odd k
else:
	numberOfSynapsesPerSegment = 3	#default: 3
	segmentActivationThreshold = 2	#default: 2 #allowing for 1 non inhibited mismatch redundancy
	useActiveBias = False
	numberNeuronsGeneratedPerSample = numberNeuronsGeneratedPerSample*2
	useDefaultNumLayersParam = False	#disable to increase number of layers

if(useInitOrigParam):
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = False
	encodeDatasetBoolValuesAs1Bit = False
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	
	datasetEqualiseClassSamples = False	
	datasetEqualiseClassSamplesTest = False	
	useMultipleTrainEpochs = False
	limitOutputConnectionsBasedOnPrevelanceAndExclusivity = False
else:
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = True
	encodeDatasetBoolValuesAs1Bit = True
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	#use output connections only from last hidden layer to output neurons
	datasetEqualiseClassSamples = True	#default: True		#optional - advantage depends on dataset class distribution
	datasetEqualiseClassSamplesTest = False	#default: False	
	useMultipleTrainEpochs = True
	limitOutputConnectionsBasedOnPrevelanceAndExclusivity = False	#limit output connectivity to prevelant class exclusive hidden neurons (used to prune network output connections and unused hidden neuron segments)
	if(limitOutputConnectionsBasedOnPrevelanceAndExclusivity):
		limitOutputConnectionsPrevelanceMin = 5	#minimum connection weight to be retained after pruning
		useBinaryOutputConnections = False	#use integer weighted connections to calculate prevelance before prune
		useBinaryOutputConnectionsEffective = True	#after prune, output connection weights are set to 0 or 1

trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/EISANIpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEISANI'


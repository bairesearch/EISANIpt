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

import math

debugEISANIdynamicUsage = False	#print neuronSegmentAssignedMask available.numel() - number of linear layer hidden features used
debugEISANIfractionActivated = False	#print fractionActive of each layer
debugMeasureClassExclusiveNeuronRatio = False	#measure ratio of a) class (output neuron) exclusive hidden neurons to b) non class (output neuron) exclusive hidden neurons
debugMeasureRatioOfHiddenNeuronsWithOutputConnections = False	#measure ratio of hidden neurons with output connections to those without output connections
debugEISANICNNdynamicallyGenerateLinearInputFeatures = False	#print nextLinearCol - number of linear layer input encoding features used
debugLimitOutputConnections = True

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
	EISANICNNcontinuousVarEncodingNumBits = 1	#default: 1	#8	#number of bits to encode image pixels
	encodedFeatureSizeDefault = 12800000*math.ceil(EISANICNNcontinuousVarEncodingNumBits/2)	#input linear layer encoded features are dynamically generated from historic active neurons in final CNN layer	#configured for numberOfConvlayers=2
	EISANICNNinputChannelThreshold = 0.5
	EISANICNNoptimisationSparseConv = True	#default: True	#only apply convolution to channels with at least 1 on bit
	EISANICNNoptimisationAssumeInt8 = False	#default: False	#if True; cnn operations (conv2d/maxpool2d) are not currently implemented on CuDNN, so will still be temporarily converted to float
	if(EISANICNNoptimisationSparseConv):
		EISANICNNdynamicallyGenerateLinearInputFeatures = True	#default: True	#input linear layer encoded features are dynamically generated from historic active neurons in final CNN layer
	else:
		EISANICNNdynamicallyGenerateLinearInputFeatures = False	#mandatory: False	#EISANICNNdynamicallyGenerateLinearInputFeatures requires EISANICNNoptimisationSparseConv and numberOfConvlayers > 1
	trainNumberOfEpochsHigh = False	#default: False
else:
	useTabularDataset = True

useInhibition = True	#default: True	#if False: only use excitatory neurons/synapses
useDynamicGeneratedHiddenConnections = True	#dynamically generate hidden neuron connections (else use randomly initialised hidden connections)
if(useDynamicGeneratedHiddenConnections):
	useDynamicGeneratedHiddenConnectionsVectorised = True	#execute entire batch simultaneously
useEIneurons = False	#use separate excitatory and inhibitory neurons (else use excitatory and inhibitory connections/synapses)
useSparseMatrix = True	#use sparse tensors to store connections (else use dense tensors)	#mandatory for any reasonably sized EISANI network
useGrayCode = True	#use graycode to encode continuous vars into binary (else use thermometer encoding)
continuousVarMin = 0.0	#sync with datasetNormaliseMinMax
continuousVarMax = 1.0	#sync with datasetNormaliseMinMax

numberOfSegmentsPerNeuron = 1 #number of segments per neuron
segmentIndexToUpdate = 0 # Placeholder	#TODO: update segmentIndexToUpdate based on dataset index. Using 0 as a placeholder.

targetActivationSparsityFraction = 0.1	#ideal number of neurons simultaneously active per layer
if(useDefaultNumNeuronsParam):
	continuousVarEncodingNumBits = 8	#default: 8	#number of bits to encode a continuous variable to	#for higher train performance numberNeuronsGeneratedPerSample should be increased (eg 16), however this requires a high numberNeuronsGeneratedPerSample+hiddenLayerSizeSANI to capture the larger number of input variations
	numberNeuronsGeneratedPerSample = 5	#default: 5	#heuristic: hiddenLayerSizeSANI//numberOfSynapsesPerSegment  	#for higher train performance numberNeuronsGeneratedPerSample should be increased substantially (eg 50), however this assigns a proportional number of additional neurons to the network (limited by hiddenLayerSizeSANI)
else:
	continuousVarEncodingNumBits = 16	#default: 16
	numberNeuronsGeneratedPerSample = 50
if(useEIneurons):
	EIneuronsMatchComputation = False	#default: False	#an additional layer is required to perform the same computation as !useEIneurons
	#if(EIneuronsMatchComputation): numberNeuronsGeneratedPerSample *= 2
if(useDynamicGeneratedHiddenConnections):
	hiddenLayerSizeSANIbase = numberNeuronsGeneratedPerSample	#heuristic: >> hiddenLayerSizeTypical * continuousVarEncodingNumBits
	initialiseSANIlayerWeightsUsingCPU = False
else:
	hiddenLayerSizeSANI = 5120000	#default: 1280000*100 with batchSize //= numberOfLayers	#large randomly initialised sparse EISANI network width 
	initialiseSANIlayerWeightsUsingCPU = False 	#optional
	
if(useDefaultSegmentSizeParam):
	numberOfSynapsesPerSegment = 5	#default: 5	#exp: 15	#number of input connections per neuron "segment"; there is 1 segment per neuron in this implementation
	segmentActivationThreshold = 3	#default: 3; allowing for 1 inhibited mismatch redundancy or 2 non inhibited mismatch redundancy	#minimum net activation required for neuron to fire (>= value), should be less than numberOfSynapsesPerSegment	#total neuron z activation expected from summation of excitatory connections to previous layer neurons
	useActiveBias = True	#bias positive (ceil) for odd k
	if(not useInhibition):
		numberOfSynapsesPerSegment = numberOfSynapsesPerSegment-1
else:
	numberOfSynapsesPerSegment = 3	#default: 3
	segmentActivationThreshold = 2	#default: 2 #allowing for 1 non inhibited mismatch redundancy
	useActiveBias = False
	numberNeuronsGeneratedPerSample = numberNeuronsGeneratedPerSample*2
	useDefaultNumLayersParam = False	#disable to increase number of layers

if(useInitOrigParam):
	useBinaryOutputConnections = True	#use binary weighted connections from hidden neurons to output neurons
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = False
	encodeDatasetBoolValuesAs1Bit = False
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	
	datasetEqualiseClassSamples = False	
	datasetEqualiseClassSamplesTest = False	
	useMultipleTrainEpochsSmallDatasetsOnly = True #emulate original dataset repeat x10 and epochs x10 for 4 small datasets (titanic, red-wine, breast-cancer-wisconsin, new-thyroid)
	limitOutputConnections = False
	useBinaryOutputConnectionsEffective = False
	useOutputConnectionsNormalised = False
else:
	useBinaryOutputConnections = False	#use binary weighted connections from hidden neurons to output neurons
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = True
	encodeDatasetBoolValuesAs1Bit = True
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	#use output connections only from last hidden layer to output neurons
	datasetEqualiseClassSamples = True	#default: True		#optional - advantage depends on dataset class distribution
	datasetEqualiseClassSamplesTest = False	#default: False	
	useMultipleTrainEpochsSmallDatasetsOnly = False
	limitOutputConnectionsBasedOnPrevalence = False	#optional	#limit output connectivity to prevelant hidden neurons (used to prune network output connections and unused hidden neuron segments)
	limitOutputConnectionsBasedOnExclusivity = False	#experimental	#limit output connectivity to class exclusive hidden neurons (used to prune network output connections and unused hidden neuron segments)
	limitOutputConnectionsBasedOnAccuracy = False	#optional	#limit output connectivity to accurate hidden neurons; associated output class predictions observed during training (used to prune network output connections and unused hidden neuron segments)
	useBinaryOutputConnectionsEffective = False
	if(limitOutputConnectionsBasedOnPrevalence or limitOutputConnectionsBasedOnExclusivity or limitOutputConnectionsBasedOnAccuracy):
		limitOutputConnections = True
		limitOutputConnectionsPrevalenceMin = 5	#minimum connection weight to be retained after pruning (unnormalised)
		limitOutputConnectionsAccuracyMin = 0.5	#minimum train prediction accuracy to be retained after pruning
		limitOutputConnectionsSoftmaxWeightMin = 0.5	#minimum hidden neuron normalised+softmax output connection weight to accept as predictive of output class y (ie accurate=True)
		if(useBinaryOutputConnections):
			useBinaryOutputConnections = False	#use integer weighted connections to calculate prevelance before prune
			useBinaryOutputConnectionsEffective = True	#after prune, output connection weights are set to 0 or 1
	else:
		limitOutputConnections = False
	if(useBinaryOutputConnections):
		useOutputConnectionsNormalised = False
	else:
		useOutputConnectionsNormalised = True	#uses tanh to normalise output connection weights between 0 and 1
		useOutputConnectionsNormalisationRange = 1.0	#divide tanh input by useOutputConnectionsNormalisationRange

recursiveLayers = False	#default: False
if(recursiveLayers): 
	recursiveSuperblocks = False	#default: False
	if(recursiveSuperblocks):
		recursiveSuperblocksNumber = 2
	else:
		recursiveSuperblocksNumber = 1
else:
	recursiveSuperblocksNumber = 1
	
trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/EISANIpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEISANI'


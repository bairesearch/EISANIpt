"""EIANNpt_EIANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EIANNpt EIANN globalDefs

"""

#debug parameters:
debugUsePositiveWeightsVerify = False
debugSmallNetwork = False
debugSanityChecks = False

#association matrix parameters:
trainInactiveNeurons = True
#associationMatrixMethod="useInputsAndOutputs"
associationMatrixMethod="useInputsAndWeights"
	
#activation function parameters:
#trainThreshold="positive"
trainThreshold="zero"	#orig	#treats neural network as a progressive input factorisation process
if(trainThreshold=="positive"):
	activationFunction="positiveMid"
	trainThresholdPositiveValue = 0.1
elif(trainThreshold=="zero"):
	activationFunction="positiveAll"	#orig
if(activationFunction=="positiveAll"):
	activationFunctionType = "relu"
elif(activationFunction=="positiveMid"):
	activationFunctionType = "clippedRelu"
	clippedReluMaxValue = trainThresholdPositiveValue*2
	
#inhibitory layer parameters:
inhibitoryNeuronInitialisationMethod="useInputActivations" #sameAsExcitatoryNeurons	#network is symmetrical
#inhibitoryNeuronInitialisationMethod="intermediaryInterneuron"	#treat inhibitory layers as intermediary (~interneuron) layers between excitatory layers
#inhibitoryNeuronInitialisationMethod="firstHiddenLayerExcitatoryInputOnly"	#network is symmetrical but requires first hidden layer activation to be renormalised (e.g. with top k),

#EI layer paramters:
#consider adjust the learning algorithm hebbian matrix application (to prevent collapse of I/E neuron weights to same values)
useDifferentEIlayerSizes = True	#ensure E I layer sizes differ to prevent collapse (to same function)	#this is a necessary (but insufficient condition?) to prevent collapse
EIANNlocalLearningApplyError = True
EIANNassociationMatrixBatched = False
if(EIANNlocalLearningApplyError):
	EIANNassociationMatrixBatched = True
if(inhibitoryNeuronInitialisationMethod=="intermediaryInterneuron"):
	hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix = True	#required as first E layer e/i input matrices are of different shapes (input vs hidden size)
else:
	if(useDifferentEIlayerSizes):
		hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix = True	#required as E/I layer sizes differ
	else:
		hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix = True	#optional
useSignedWeights = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = False	#clamp entire model weights to be positive (rather than per layer)

#loss function paramters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

#train paramters:
trainLastLayerOnly = True	#required	#EIANN
if(trainLastLayerOnly):
	EIANNlocalLearning = True
	#normaliseActivationSparsity = False
	if(EIANNlocalLearning):
		EIANNlocalLearningNeuronActiveThreshold = 0.0	#minimum activation level for neuron to be considered active	#CHECKTHIS
		EIANNlocalLearningRate = 0.001	#0.01	#default: 0.001	#CHECKTHIS
		EIANNlocalLearningBias = False	#bias learning towards most signficant weights

#network hierarchy parameters: 
if(trainLastLayerOnly):
	#override ANNpt_globalDefs default model parameters;
	batchSize = 64
	numberOfLayers = 4	#CHECKTHIS
	if(useDifferentEIlayerSizes):
		hiddenLayerSizeE = 10	
		hiddenLayerSizeI = 7
	else:
		hiddenLayerSize = 10	#not used
		hiddenLayerSizeE = hiddenLayerSize
		hiddenLayerSizeI = hiddenLayerSize
		
#initialisation parameters:
normaliseActivationSparsity = False
useCustomBiasInitialisation = True	#required to set all biases to 0


workingDrive = '/large/source/ANNpython/EIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEIANN'


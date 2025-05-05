"""EIANNpt_EIANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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
debugUsePositiveWeightsVerify = True
debugSmallNetwork = False
debugSanityChecks = True
debugEIinputs = False


#training update implementation parameters:
trainingUpdateImplementation = "backprop"	#single layer backprop
#trainingUpdateImplementation = "hebbian"	#orig	#custom approximation of single layer backprop

#optimisation function parameters:
if(trainingUpdateImplementation=="backprop"): 
	optimisationFunctionType = "vicreg"	#vicreg function for i/e inputs	#currently requires trainingUpdateImplementation = "backprop"
	#optimisationFunctionType = "similarity"	#orig #simple similarity function for i/e inputs
else:
	optimisationFunctionType = "similarity"	#orig #simple similarity function for i/e inputs
if(trainingUpdateImplementation == "backprop"):
	if(optimisationFunctionType=="similarity"):
		vicregSimilarityLossOnly = True
	elif(optimisationFunctionType=="vicreg"):
		vicregSimilarityLossOnly = False
	debugVICRegLoss = False
	partiallyAlignLayer = False
	trainMostAlignedNeurons = False
	if(optimisationFunctionType == "vicreg"):
		usePairedDataset = False	#formal vicreg algorithm uses paired dataset, EIANN does not
		lambdaHyperparameter = 1.0 #invariance coefficient	#base condition > 1
		muHyperparameter = 1.0	#invariance coefficient	#base condition > 1
		nuHyperparameter = 1.0 #covariance loss coefficient	#set to 1
	elif(optimisationFunctionType == "similarity"):
		lambdaHyperparameter = 1.0 #invariance coefficient	#base condition > 1
	trainLocal = True	#local learning rule	#required
else:
	trainLocal = False	#optional	#orig=False

#association matrix parameters:
trainInactiveNeurons = False	#if False, e/i weights will continue to increase (but cancel each other out: resulting distribution remains normalised)
#associationMatrixMethod="useInputsAndOutputs"
associationMatrixMethod="useInputsAndWeights"

#activation function parameters:
invertActivationFunctionForEIneurons = True	#E and I neurons are activated by inputs being above and below threshold (0) respectively
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
inhibitoryNeuronOutputPositive = False	#orig: False	#I and E neurons detect presence and absence of stimuli, but both produce excitatory (ie positive) output
if(inhibitoryNeuronOutputPositive):
	inhibitoryNeuronSwitchActivation = True	#required
else:
	inhibitoryNeuronSwitchActivation = False	#optional	#orig: False	#E predicts on antecedents, I predicts off antecedents
if(inhibitoryNeuronSwitchActivation):
	datasetNormaliseStdAvg = True	#normalise based on std and mean (~-1.0 to 1.0)
	datasetNormaliseMinMax = False
inhibitoryNeuronInitialisationMethod="useInputActivations" #sameAsExcitatoryNeurons	#network is symmetrical
#inhibitoryNeuronInitialisationMethod="intermediaryInterneuron"	#treat inhibitory layers as intermediary (~interneuron) layers between excitatory layers
#inhibitoryNeuronInitialisationMethod="firstHiddenLayerExcitatoryInputOnly"	#network is symmetrical but requires first hidden layer activation to be renormalised (e.g. with top k),
if(inhibitoryNeuronOutputPositive):
	assert inhibitoryNeuronInitialisationMethod=="useInputActivations"

#EI layer paramters:
#consider adjust the learning algorithm hebbian matrix application (to prevent collapse of I/E neuron weights to same values)
if(inhibitoryNeuronOutputPositive):
	useDifferentEIlayerSizes = False	#required
else:
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
if(inhibitoryNeuronOutputPositive):
	useSignedWeights = False	#required
else:
	useSignedWeights = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = False	#clamp entire model weights to be positive (rather than per layer)

#loss function paramters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

#train paramters:
EIANNlocalLearning = True
#normaliseActivationSparsity = False
if(EIANNlocalLearning):
	EIANNlocalLearningNeuronActiveThreshold = 0.0	#minimum activation level for neuron to be considered active	#CHECKTHIS
	EIANNlocalLearningRate = 0.001	#0.01	#default: 0.001	#CHECKTHIS
	EIANNlocalLearningBias = False	#bias learning towards most signficant weights

#network hierarchy parameters: 
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
	if(inhibitoryNeuronOutputPositive):
		assert hiddenLayerSize%2 == 0

#initialisation parameters:
normaliseActivationSparsity = False
#useCustomWeightInitialisation = True	#CHECKTHIS
useCustomBiasInitialisation = True	#required to set all biases to 0	#initialise biases to zero
if(useCustomBiasInitialisation):
	if(trainingUpdateImplementation=="backprop"): 
		useCustomBiasNoTrain = True

workingDrive = '/large/source/ANNpython/EIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEIANN'


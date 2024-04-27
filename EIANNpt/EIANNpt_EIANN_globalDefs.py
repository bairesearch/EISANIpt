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

debugSmallNetwork = False

activationFunctionType = "relu"

EIANNlocalLearningApplyError = True

firstHiddenLayerExcitatoryInputOnly = False	#True: network is more symmetrical but requires first hidden layer activation to be renormalised (e.g. with top k), False: treat inhibitory layers as intermediary (~interneuron) layers between excitatory layers
	#!firstHiddenLayerExcitatoryInputOnly; consider adjust the learning algorithm hebbian matrix application (to prevent collapse of I/E neuron weights to same values)
if(firstHiddenLayerExcitatoryInputOnly):
	hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix = False	#optional
else:
	hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix = True	#true	#required as first layer matrices are of different shapes

useSignedWeights = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = False	#clamp entire model weights to be positive (rather than per layer)
useInbuiltCrossEntropyLossFunction = True	#required
		
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

trainLastLayerOnly = True	#required	#EIANN
if(trainLastLayerOnly):
	EIANNlocalLearning = True
	#normaliseActivationSparsity = False
	if(EIANNlocalLearning):
		EIANNlocalLearningNeuronActiveThreshold = 0.0	#minimum activation level for neuron to be considered active	#CHECKTHIS
		EIANNlocalLearningRate = 0.001	#0.01	#default: 0.001	#CHECKTHIS
		EIANNlocalLearningBias = False	#bias learning towards most signficant weights
		
if(trainLastLayerOnly):
	#override ANNpt_globalDefs default model parameters;
	batchSize = 64
	numberOfLayers = 4	#CHECKTHIS
	hiddenLayerSize = 10	#CHECKTHIS

	normaliseActivationSparsity = True	#increases performance


workingDrive = '/large/source/ANNpython/EIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEIANN'


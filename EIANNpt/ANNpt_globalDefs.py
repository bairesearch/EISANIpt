"""ANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt globalDefs

"""

#algorithm selection
useAlgorithmVICRegANN = False
useAlgorithmAUANN = False
useAlgorithmLIANN = False
useAlgorithmLUANN = False
useAlgorithmLUOR = False
useAlgorithmSANIOR = False
useAlgorithmEIANN = True
useAlgorithmEIOR = False

#initialise (dependent vars);
usePairedDataset = False
datasetNormalise = False
datasetRepeat = False
datasetShuffle = False	#automatically performed by generateVICRegANNpairedDatasets
datasetOrderByClass = False	#automatically performed by generateVICRegANNpairedDatasets
dataloaderShuffle = True
dataloaderMaintainBatchSize = True
dataloaderRepeat = False

optimiserAdam = True

#initialise (dependent vars);
useCustomWeightInitialisation = False
useCustomBiasInitialisation = False
useSignedWeights = False
usePositiveWeightsClampModel = False

debugSmallBatchSize = False	#small batch size for debugging matrix output
debugDataNormalisation = False

trainLocal = False
trainGreedy = False
trainIndividialSamples = False

useLinearSublayers = False	#use multiple independent sublayers per linear layer	#optional
if(useLinearSublayers):
	linearSublayersNumber = 10
else:
	linearSublayersNumber = 1
	
debugSmallNetwork = False
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1
else:
	batchSize = 64
	numberOfLayers = 4
	hiddenLayerSize = 10
	trainNumberOfEpochs = 10

#initialise (dependent vars);
inputLayerInList = True
outputLayerInList = True
useCNNlayers = False
thresholdActivations = False
debugPrintActivationOutput = False
simulatedDendriticBranches = False
activationFunctionType = "relu"
trainLastLayerOnly = False
normaliseActivationSparsity = False
debugUsePositiveWeightsVerify = False

useTabularDataset = False
useImageDataset = False
if(useAlgorithmVICRegANN):
	from VICRegANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmAUANN):
	from LREANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLIANN):
	from LIANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUANN):
	from LUANNpt_LUANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUOR):
	from LUANNpt_LUOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmSANIOR):
	from LUANNpt_SANIOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmEIANN):
	from EIANNpt_EIANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmEIOR):
	from EIANNpt_EIOR_globalDefs import *
	useImageDataset = True
	
import torch as pt

useLovelyTensors = True
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pt.set_printoptions(profile="full")
	pt.set_printoptions(sci_mode=False)
	
#pt.autograd.set_detect_anomaly(True)

stateTrainDataset = True
stateTestDataset = True

if(useCustomWeightInitialisation):
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations

#initialise (dependent vars);
datasetReplaceNoneValues = False
datasetNormaliseClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False


dataloaderRepeatSampler = False	#initialise (dependent var)
dataloaderRepeatLoop = False	#initialise (dependent var)	#legacy (depreciate)
if(useTabularDataset):
	#datasetName = 'tabular-benchmark'
	#datasetName = 'blog-feedback'
	#datasetName = 'titanic'
	#datasetName = 'red-wine'
	#datasetName = 'breast-cancer-wisconsin'
	#datasetName = 'diabetes-readmission'
	datasetName = 'new-thyroid'
	if(datasetName == 'tabular-benchmark'):
		datasetNameFull = 'inria-soda/tabular-benchmark'
		classFieldName = 'class'
		trainFileName = 'clf_cat/albert.csv'
		testFileName = 'clf_cat/albert.csv'
		datasetNormalise = True
	elif(datasetName == 'blog-feedback'):
		datasetNameFull = 'wwydmanski/blog-feedback'
		classFieldName = 'target'
		trainFileName = 'train.csv'
		testFileName = 'test.csv'
		datasetNormalise = True
		datasetNormaliseClassValues = True	#int: not contiguous	#CHECKTHIS
	elif(datasetName == 'titanic'):
		datasetNameFull = 'victor/titanic'
		classFieldName = '2urvived'
		trainFileName = 'train_and_test2.csv'	#train
		testFileName = 'train_and_test2.csv'	#test
		datasetReplaceNoneValues = True
		datasetNormalise = True
	elif(datasetName == 'red-wine'):
		datasetNameFull = 'lvwerra/red-wine'
		classFieldName = 'quality'
		trainFileName = 'winequality-red.csv'
		testFileName = 'winequality-red.csv'
		datasetNormaliseClassValues = True	#int: not start at 0
		datasetNormalise = True
	elif(datasetName == 'breast-cancer-wisconsin'):
		datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
		classFieldName = 'diagnosis'
		trainFileName = 'breast_cancer.csv'
		testFileName = 'breast_cancer.csv'
		datasetReplaceNoneValues = True
		datasetNormaliseClassValues = True	#string: B/M	#requires conversion of target string B/M to int
		datasetNormalise = True
	elif(datasetName == 'diabetes-readmission'):
		datasetNameFull = 'imodels/diabetes-readmission'
		classFieldName = 'readmitted'
		trainFileName = 'train.csv'
		testFileName = 'test.csv'	
		datasetNormalise = True
	elif(datasetName == 'new-thyroid'):
		classFieldName = 'class'
		trainFileName = 'new-thyroid.csv'
		testFileName = 'new-thyroid.csv'
		datasetLocalFile = True	
		datasetNormalise = True
		datasetNormaliseClassValues = True
		trainNumberOfEpochs = 100
		numberOfLayers = 4
		hiddenLayerSize = 15	#5
		datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
		if(datasetRepeat):
			datasetRepeatSize = 10
		dataloaderRepeat = True
	#elif ...

	if(dataloaderRepeat):
		dataloaderRepeatSize = 10	#number of repetitions
		dataloaderRepeatSampler = True
		dataloaderRepeatLoop = False	#legacy (depreciate)
		if(dataloaderRepeatSampler):
			dataloaderRepeatSamplerCustom = False	#no tqdm visualisation
		
	if(datasetNormalise):
		datasetNormaliseMinMax = True	#normalise between 0.0 and 1.0
		datasetNormaliseStdAvg = False	#normalise based on std and mean (~-1.0 to 1.0)


if(debugSmallBatchSize):
	batchSize = 10

printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10


useInbuiltCrossEntropyLossFunction = True	#required
if(useSignedWeights):
	if(not useAlgorithmEIANN and not useAlgorithmEIOR):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required

learningRate = 0.005	#0.005	#0.0001


relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE
import os
if(os.path.isdir('user')):
	from user.user_globalDefs import *

modelSaveNumberOfBatches = 100	#resave model after x training batches

dataFolderName = 'data'
modelFolderName = 'model'
if(relativeFolderLocations):
	dataPathName = dataFolderName
	modelPathName = modelFolderName
else:	
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
modelPathNameBase = modelPathName
modelPathNameFull = getModelPathNameFull(modelPathNameBase, modelName)
	
def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = pt.cuda.get_device_properties(0).total_memory
	memory_reserved = pt.cuda.memory_reserved(0)
	memory_allocated = pt.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()

device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')


	

"""ANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

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
useAlgorithmEIANN = False
useAlgorithmEIOR = False
useAlgorithmEISANI = True
useAlgorithmAEANN = False
useAlgorithmFFANN = False

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
useCustomBiasNoTrain = False
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
	

#default network hierarchy parameters (overwritten by specific dataset defaults): 
warmupEpochs = 0	#default: 0
learningRate = 0.005	#0.005	#0.0001
momentum = 0.0     #default: 0.0
weightDecay  = 0.0 #default: 0.0
batchSize = 64	 #default: 64	#debug: 2
numberOfLayers = 4	#default: 4	#counts hidden and output layers (not input layer)
hiddenLayerSize = 10	#default: 10
trainNumberOfEpochs = 10	#default: 10
numberOfConvlayers = None
CNNhiddenLayerSize = None
useLearningRateScheduler = False
batchNormFC = False	#optional	#batch norm for fully connected layers
dropout = False
dropoutProb = 0.5 	#default: 0.5	#orig: 0.3

#initialise (dependent vars);
debugSmallNetwork = False
trainNumberOfEpochsHigh = False
inputLayerInList = True
outputLayerInList = True
useCNNlayers = False
useCNNlayers2D = True
thresholdActivations = False
debugPrintActivationOutput = False
simulatedDendriticBranches = False
activationFunctionType = "relu"
trainLastLayerOnly = False
normaliseActivationSparsity = False
debugUsePositiveWeightsVerify = False
datasetNormaliseMinMax = True	#normalise between 0.0 and 1.0
datasetNormaliseStdAvg = False	#normalise based on std and mean (~-1.0 to 1.0)
supportSkipLayers = False
supportSkipLayersResidual = False

useInbuiltCrossEntropyLossFunction = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required

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
elif(useAlgorithmEISANI):
	from EIANNpt_EISANI_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmAEANN):
	from AEANNpt_AEANN_globalDefs import *
	#useTabularDataset/useImageDataset is defined by AEANNpt_AEANN_globalDefs
elif(useAlgorithmFFANN):
	from AEANNpt_FFANN_globalDefs import *
	useTabularDataset = True
	
import torch as pt

useLovelyTensors = False
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
datasetConvertClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetConvertFeatureValues = False	#reformat feature values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False
datasetSpecifyDataFiles = True	#specify data file names in dataset (else automatically selected by huggingface)
datasetHasTestSplit = True
datasetHasSubsetType = False

datasetLocalFileOptimise = False
datasetCorrectMissingValues = False	
datasetConvertClassTargetColumnFloatToInt = False
dataloaderRepeatSampler = False	
dataloaderRepeatLoop = False		#legacy (depreciate)

if(useTabularDataset):
	#datasetName = 'tabular-benchmark'	#expected test accuracy: ~63%
	#datasetName = 'blog-feedback'	#expected test accuracy: ~70%
	datasetName = 'titanic'	#expected test accuracy: ~90%
	#datasetName = 'red-wine'	#expected test accuracy: ~58%
	#datasetName = 'breast-cancer-wisconsin'	#expected test accuracy: ~91%
	#datasetName = 'diabetes-readmission'	#expected test accuracy: ~61%
	#datasetName = 'banking-marketing'	#expected test accuracy: ~85%	#third party benchmark accuracy: ~79.1%
	#datasetName = 'adult_income_dataset'	#expected test accuracy: ~86% 	#third party benchmark accuracy: ~85.8%
	#datasetName = 'covertype'	#expected test accuracy: ~%92 (10 epochs) 	#third party benchmark accuracy: ~97.1%
	#datasetName = 'higgs'	#expected test accuracy: ~70%	#third party benchmark accuracy: 73.8%	#https://archive.ics.uci.edu/dataset/280/higgs
	#datasetName = 'new-thyroid'	#expected test accuracy: ~95%	https://archive.ics.uci.edu/dataset/102/thyroid+disease
	if(datasetName == 'tabular-benchmark'):
		datasetNameFull = 'inria-soda/tabular-benchmark'
		classFieldName = 'class'
		trainFileName = 'clf_cat/albert.csv'
		datasetHasTestSplit = False
		datasetNormalise = True
		#datasetShuffle = True	#raw dataset is not shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001
		hiddenLayerSize = 64	#default: 64	#orig: 10
		numberOfLayers = 4	#default: 4
		trainNumberOfEpochs = 1	#default: 1
	elif(datasetName == 'blog-feedback'):
		datasetNameFull = 'wwydmanski/blog-feedback'
		classFieldName = 'target'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		#datasetConvertClassValues = True	#int: not contiguous	#alternate method (slower)
		datasetConvertClassTargetColumnFloatToInt = True	#int: not contiguous
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 144	#default: 144	#orig: 144	#old 800	#144=288/2 [input features padded / 2]
		trainNumberOfEpochs = 1	#default: 1
	elif(datasetName == 'titanic'):
		datasetNameFull = 'victor/titanic'
		classFieldName = '2urvived'
		datasetSpecifyDataFiles = False
		datasetReplaceNoneValues = True
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 128	#default: 128	#orig: 100
		trainNumberOfEpochs = 10	#default: 10
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'red-wine'):
		datasetNameFull = 'lvwerra/red-wine'
		classFieldName = 'quality'
		trainFileName = 'winequality-red.csv'
		datasetHasTestSplit = False
		datasetConvertClassValues = True	#int: not start at 0
		datasetNormalise = True
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4	#external benchmark: 4
		hiddenLayerSize = 128	#default: 128	#orig: 100	#external benchmark; 64/128
		trainNumberOfEpochs = 10	#default: 10	#note train accuracy continues to increase (overfit) with increasing epochs
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'breast-cancer-wisconsin'):
		datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
		classFieldName = 'diagnosis'
		trainFileName = 'breast_cancer.csv'
		datasetHasTestSplit = False
		datasetReplaceNoneValues = True
		datasetConvertClassValues = True	#string: B/M
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 32	#default: 32	#orig: 20	#old: 100
		trainNumberOfEpochs = 10	#default: 10
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'diabetes-readmission'):
		datasetNameFull = 'imodels/diabetes-readmission'
		classFieldName = 'readmitted'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		learningRate = 0.005
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 304	#default: 304	#orig: 10, 256
		trainNumberOfEpochs = 1
	elif(datasetName == 'banking-marketing'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetNameFull = 'Andyrasika/banking-marketing'
		classFieldName = 'y'
		datasetConvertClassValues = True	#string: yes/no
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 5	#default: 5
		hiddenLayerSize = 128	#default: 128
		trainNumberOfEpochs = 1
	elif(datasetName == 'adult_income_dataset'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetNameFull = 'meghana/adult_income_dataset'
		classFieldName = 'income'
		datasetConvertClassValues = True	#string: <=50K/>50K
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 256	#default: 256
		trainNumberOfEpochs = 10	
	elif(datasetName == 'covertype'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetHasSubsetType = True
		datasetSubsetName = 'covertype'
		datasetNameFull = 'mstz/covertype'
		classFieldName = 'cover_type'
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 6	#default: 6
		hiddenLayerSize = 512	#default: 512
		#numberOfLayers = 2
		#hiddenLayerSize = 512*4
		trainNumberOfEpochs = 10		#train performance increases with higher epochs
		trainNumberOfEpochsHigh = False	#orig: True
	elif(datasetName == 'higgs'):
		datasetLocalFile = True		#manually download higgs.zip:HIGGS.csv from https://archive.ics.uci.edu/dataset/280/higgs
		#echo "class,lepton_pT,lepton_eta,lepton_phi,missing_energy_magnitude,missing_energy_phi,jet1pt,jet1eta,jet1phi,jet1b,jet2pt,jet2eta,jet2phi,jet2b,jet3pt,jet3eta,jet3phi,jet3b,jet4pt,jet4eta,jet4phi,jet4b,m_jj,m_jjj,m_lv,m_jlv,m_bb,m_wbb,m_wwbb" | cat - HIGGS.csv > HIGGSwithHeader.csv
		classFieldName = 'class'
		trainFileName = 'HIGGSwithHeader.csv'
		datasetHasTestSplit = False
		datasetConvertClassTargetColumnFloatToInt = True
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 4
		#numberOfLayers = 2
		hiddenLayerSize = 256	#default: 256
		batchSize = 64		#4096	#default: 64
		trainNumberOfEpochs = 10
		datasetLocalFileOptimise = False	#default: False
	elif(datasetName == 'new-thyroid'):
		classFieldName = 'class'
		trainFileName = 'new-thyroid.csv'	#manually download thyroid+disease.zip:new-thyroid.data from https://archive.ics.uci.edu/dataset/102/thyroid+disease
		datasetHasTestSplit = False
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		datasetLocalFile = True	
		datasetNormalise = True
		datasetConvertClassValues = True
		learningRate = 0.005	#default: 0.005
		numberOfLayers = 3	#default: 3	#orig: 2, 4
		hiddenLayerSize = 16	#default: 16	#orig: 4, 10
		#batchSize = 1
		trainNumberOfEpochs = 100	#default: 100
		datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
		if(datasetRepeat):
			datasetRepeatSize = 100	#for batchSize ~= 64
	#elif ...

	if(datasetLocalFileOptimise):
		datasetShuffle = True	#default: True	#required for high dataloader initialisation efficiency with large datasets
		dataloaderShuffle = False	#default: False	#required for high dataloader initialisation efficiency with large datasets
		disableDatasetCache = True	#default: False #requires high CPU ram	#prevents large cache from being created on disk #only suitable with datasetLocalFile
	else:
		disableDatasetCache = False
		
	if(dataloaderRepeat):
		dataloaderRepeatSize = 10	#number of repetitions
		dataloaderRepeatLoop = False	#legacy (depreciate)
		dataloaderRepeatSampler = True
		if(dataloaderRepeatSampler):
			dataloaderRepeatSamplerCustom = False	#no tqdm visualisation
			assert not dataloaderShuffle	#dataloaderShuffle is not supported by dataloaderRepeatSampler
elif(useImageDataset):
	#currently assume CIFAR-10 dataset	#expected test accuracy: ~91%
	warmupEpochs = 5 	#default: 5	#orig: 0
	learningRate = 0.001	#default: 0.001 (or 0.01)	#orig: 0.005
	momentum = 0.9     #default: 0.9	#orig: 0.0
	weightDecay  = 5e-4    #default: 5e-4	#orig: 0.0
	batchSize = 128	 #default: 128	#orig: 64
	numberOfConvlayers = 6	#rest will be linear	#orig: 2	#default: 2, 4, 6
	numberOfLayers = numberOfConvlayers+3	#counts hidden and output layers (not input layer)
	#numberOfLayers = 2
	#numberOfConvlayers = 0
	numberInputImageChannels = 3
	CNNhiddenLayerSize = 3*32*32 * 4	#default: CIFAR-10 image size = 3*32*32=3072
	if(numberOfConvlayers >= 4):
		CNNconvergeEveryEvenLayer = True	#default: True for numberOfConvlayers 4 or 6	#orig: False
		assert numberOfConvlayers%2 == 0
	else:
		CNNconvergeEveryEvenLayer = False
	hiddenLayerSize = 1024
	disableDatasetCache = False
	imageDatasetAugment = True
	if(imageDatasetAugment):
		trainNumberOfEpochs = 100	#default: 100
	else:
		trainNumberOfEpochs = 10	#default: 10
	useLearningRateScheduler = True
	learningRateSchedulerStepsize = 10	#default: 30	#orig: 10	
	learningRateSchedulerGamma = 0.5	#default: 0.1	#orig: 0.5
	batchNormFC = False	#optional	#batch norm for fully connected layers
	dropout = False	#default: False
	dropoutProb = 0.5 	#default: 0.5	#orig: 0.3

if(useAlgorithmEISANI):
	trainNumberOfEpochs = 1
	
if(trainNumberOfEpochsHigh):
	trainNumberOfEpochs = trainNumberOfEpochs*4
	
if(debugSmallBatchSize):
	batchSize = 10
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1
	
printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10

datasetSplitNameTrain = 'train'
datasetSplitNameTest = 'test'
if(not datasetHasTestSplit):
	datasetTestSplitSize = 0.1

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


	

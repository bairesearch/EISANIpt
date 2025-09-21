"""EISANIpt_EISANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt excitatory inhibitory (EI) sequentially/summation activated neuronal input (SANI) network

"""

import torch as pt
from ANNpt_globalDefs import *
#from torchsummary import summary
import EISANIpt_EISANImodel
import ANNpt_data
if(useStochasticUpdates):
	import EISANIpt_EISANIstochasticUpdate


def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=False)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)	#note "numberOfFeatures" is the raw continuous var input (without x-bit encoding)	#not used
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)
	hiddenLayerSizeSANI = EISANIpt_EISANImodel.generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers, hiddenLayerSize)

	if(printEISANImodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t useTabularDataset = ", useTabularDataset)
		print("\t useImageDataset = ", useImageDataset)
		print("\t useNLPDataset = ", useNLPDataset)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetSize = ", datasetSize)
		print("\t datasetEqualiseClassSamples = ", datasetEqualiseClassSamples)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		#print("\t useDefaultNumNeuronSegmentsParam = ", useDefaultNumNeuronSegmentsParam)
		#print("\t useDefaultSegmentSizeParam = ", useDefaultSegmentSizeParam)
		print("\t useDefaultNumLayersParam = ", useDefaultNumLayersParam)
		#print("\t useInitOrigParam = ", useInitOrigParam)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t numberOfConvlayers = ", numberOfConvlayers)
		print("\t hiddenLayerSizeSANI = ", hiddenLayerSizeSANI)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t numberOfSynapsesPerSegment = ", numberOfSynapsesPerSegment)
		print("\t numberOfSegmentsPerNeuron = ", numberOfSegmentsPerNeuron)
		print("\t ---")
		print("\t useBinaryOutputConnections = ", useBinaryOutputConnections)
		print("\t useDynamicGeneratedHiddenConnections = ", useDynamicGeneratedHiddenConnections)
		print("\t useEIneurons = ", useEIneurons)
		print("\t EISANITABcontinuousVarEncodingNumBits = ", EISANITABcontinuousVarEncodingNumBits)
		print("\t numberNeuronSegmentsGeneratedPerSample = ", numberNeuronSegmentsGeneratedPerSample)
		print("\t recursiveLayers = ", recursiveLayers)
		print("\t recursiveSuperblocksNumber = ", recursiveSuperblocksNumber)
		print("\t useOutputConnectionsNormalised = ", useOutputConnectionsNormalised)
		if(limitConnections and limitOutputConnections):
			print("\t limitOutputConnectionsBasedOnPrevalence = ", limitOutputConnectionsBasedOnPrevalence)
			print("\t limitOutputConnectionsBasedOnAccuracy = ", limitOutputConnectionsBasedOnAccuracy)	
		if(useNLPDataset):
			print("\t ---")
			print("\t useNLPDataset:")	
			print("\t\t evalOnlyUsingTimeInvariance = ", evalOnlyUsingTimeInvariance)
			print("\t\t evalStillTrainOutputConnections = ", evalStillTrainOutputConnections)
			print("\t\t sequentialSANItimeInvariance = ", sequentialSANItimeInvariance)
			print("\t\t useSequentialSANIactivationStrength = ", useSequentialSANIactivationStrength)
			print("\t\t sequentialSANIoverlappingSegments = ", sequentialSANIoverlappingSegments)
			print("\t\t datasetTrainRows = ", datasetTrainRows)
			print("\t\t datasetTestRows = ", datasetTestRows)
		elif(useImageDataset):
			print("\t ---")
			print("\t useImageDataset:")	
			print("\t\t EISANICNNnumberKernelOrientations = ", EISANICNNnumberKernelOrientations)
			print("\t\t EISANITABcontinuousVarEncodingNumBitsAfterCNN = ", EISANITABcontinuousVarEncodingNumBitsAfterCNN)
		print("\t ---") 
		print("\t useStochasticUpdates = ", useStochasticUpdates)
		if(useStochasticUpdates):
			print("\t\t useStochasticUpdatesHiddenUnitLearning = ", useStochasticUpdatesHiddenUnitLearning)
			print("\t\t stochasticHiddenUpdatesPerBatch = ", stochasticHiddenUpdatesPerBatch)
			print("\t\t stochasticOutputLearningRate = ", stochasticOutputLearningRate)
			print("\t\t hiddenLayerSizeSANImultiplier = ", hiddenLayerSizeSANImultiplier)
			
		
	config = EISANIpt_EISANImodel.EISANIconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSizeSANI,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		numberOfSynapsesPerSegment = numberOfSynapsesPerSegment,
		fieldTypeList = fieldTypeList,
	)
	model = EISANIpt_EISANImodel.EISANImodel(config)
	
	print(model)

	return model


def trainOrTestModel(model, trainOrTest, x, y, optim=None, l=None, batchIndex=None, fieldTypeList=None):
	if(trainOrTest and useStochasticUpdates):
		return EISANIpt_EISANIstochasticUpdate.performStochasticUpdate(model, trainOrTest, x, y, optim, l, batchIndex, fieldTypeList)
	else:
		return model(trainOrTest, x, y, optim, l, batchIndex, fieldTypeList)

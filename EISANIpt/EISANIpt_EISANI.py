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

from ANNpt_globalDefs import *
from torchsummary import summary
import EISANIpt_EISANImodel
import ANNpt_data


def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=False)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)	#note "numberOfFeatures" is the raw continuous var input (without x-bit encoding)	#not used
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)
	hiddenLayerSizeSANI = EISANIpt_EISANImodel.generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers)

	print("Creating new model:")
	print("\t datasetName = ", datasetName)
	print("\t datasetSize = ", datasetSize)
	print("\t datasetEqualiseClassSamples = ", datasetEqualiseClassSamples)
	print("\t datasetRepeatSize = ", datasetRepeatSize)
	print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
	print("\t ---")
	print("\t batchSize = ", batchSize)
	print("\t numberOfLayers = ", numberOfLayers)
	print("\t numberOfConvlayers = ", numberOfConvlayers)
	print("\t hiddenLayerSizeSANI = ", hiddenLayerSizeSANI)
	print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
	print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
	print("\t numberOfSynapsesPerSegment = ", numberOfSynapsesPerSegment)
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

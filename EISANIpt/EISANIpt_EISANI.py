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

import EISANIpt_EISANI_globalDefs
from ANNpt_globalDefs import *
from torchsummary import summary
import EISANIpt_EISANImodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=False)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)
	if(useDynamicGeneratedHiddenConnections):
		datasetSizeRounded = round_up_to_power_of_2(datasetSize)
		hiddenLayerSizeSANI = hiddenLayerSizeSANIbase*datasetSizeRounded * trainNumberOfEpochs
	else:
		hiddenLayerSizeSANI = EISANIpt_EISANI_globalDefs.hiddenLayerSizeSANI
	print("Creating new model:")
	print("\tdatasetName = ", datasetName)
	print("\tdatasetSize = ", datasetSize)
	print("\tbatchSize = ", batchSize)
	print("\tnumberOfLayers = ", numberOfLayers)
	print("\tnumberOfConvlayers = ", numberOfConvlayers)
	print("\thiddenLayerSizeSANI = ", hiddenLayerSizeSANI)
	print("\tinputLayerSize (numberOfFeatures) = ", numberOfFeatures)
	print("\toutputLayerSize (numberOfClasses) = ", numberOfClasses)
	print("\tnumberOfSynapsesPerSegment = ", numberOfSynapsesPerSegment)
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

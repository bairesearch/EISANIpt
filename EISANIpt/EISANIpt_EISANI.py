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
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)

	print("creating new model")
	config = EISANIpt_EISANImodel.EISANIconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
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

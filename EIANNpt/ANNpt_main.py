"""ANNpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision	[required for OR]
pip install chardet
pip install cchardet

# Usage:
source activate pytorchsenv
python ANNpt_main.py

# Description:
ANNpt main - custom artificial neural network trained on tabular data

"""

import torch
from tqdm.auto import tqdm
from torch import optim

from ANNpt_globalDefs import *

if(useAlgorithmVICRegANN):
	import VICRegANNpt_VICRegANN as ANNpt_algorithm
elif(useAlgorithmAUANN):
	import LREANNpt_AUANN as ANNpt_algorithm
elif(useAlgorithmLIANN):
	import LIANNpt_LIANN as ANNpt_algorithm
elif(useAlgorithmLUANN):
	import LUANNpt_LUANN as ANNpt_algorithm
elif(useAlgorithmLUOR):
	import LUANNpt_LUOR as ANNpt_algorithm
elif(useAlgorithmSANIOR):
	import LUANNpt_SANIOR as ANNpt_algorithm
elif(useAlgorithmEIANN):
	import EIANNpt_EIANN as ANNpt_algorithm
elif(useAlgorithmEIOR):
	import EIANNpt_EIOR as ANNpt_algorithm
	
if(useSignedWeights):
	import ANNpt_linearSublayers
import ANNpt_data

#https://huggingface.co/docs/datasets/tabular_load

def main():
	dataset = ANNpt_data.loadDataset()
	if(stateTrainDataset):
		model = ANNpt_algorithm.createModel(dataset['train'])	#dataset['test'] not possible as test does not contain all classes
		processDataset(True, dataset['train'], model)
	if(stateTestDataset):
		model = loadModel()
		processDataset(False, dataset['test'], model)

def createOptimizer():
	if(optimiserAdam):
		optim = torch.optim.Adam(model.parameters(), lr=learningRate)
	else:
		optim = torch.optim.SGD(model.parameters(), lr=learningRate)
	return optim

def createOptimiser(model):
	if(trainLocal):
		if(trainIndividialSamples):
			optim = [[None for layerIndex in range(model.config.numberOfLayers) ] for sampleIndex in range(batchSize)]
			for sampleIndex in range(batchSize):
				for layerIndex in range(model.config.numberOfLayers):
					optimSampleLayer = torch.optim.Adam(model.parameters(), lr=learningRate)
					optim[sampleIndex][layerIndex] = optimSampleLayer
		else:
			optim = [None]*model.config.numberOfLayers
			for layerIndex in range(model.config.numberOfLayers):
				optimLayer = torch.optim.Adam(model.parameters(), lr=learningRate)
				optim[layerIndex] = optimLayer
	else:
		optim = torch.optim.Adam(model.parameters(), lr=learningRate)
	return optim

def processDataset(trainOrTest, dataset, model):
	if(trainOrTest):
		if(useAlgorithmEIANN and trainLocal):
			optim = []
			optim += [createOptimiser(model)]
			optim += [createOptimiser(model)]
		else:
			optim = createOptimiser(model)
		model.to(device)
		model.train()	
		numberOfEpochs = trainNumberOfEpochs
	else:
		model.to(device)
		model.eval()
		numberOfEpochs = 1
	
	if(useAlgorithmLUOR):
		ANNpt_algorithm.preprocessLUANNpermutations(dataset, model)
		
	for epoch in range(numberOfEpochs):
		if(usePairedDataset):
			dataset1, dataset2 = ANNpt_algorithm.generateVICRegANNpairedDatasets(dataset)
		
		if(trainGreedy):
			maxLayer = model.config.numberOfLayers
		else:
			maxLayer = 1
		for l in range(maxLayer):
			if(trainGreedy):
				print("trainGreedy: l = ", l)
			
			if(printAccuracyRunningAverage):
				(runningLoss, runningAccuracy) = (0.0, 0.0)

			if(dataloaderRepeatLoop):
				numberOfDataloaderIterations = dataloaderRepeatSize
			else:
				numberOfDataloaderIterations = 1
			for dataLoaderIteration in range(numberOfDataloaderIterations):
				if(useTabularDataset):
					if(usePairedDataset):
						loader = ANNpt_data.createDataLoaderTabularPaired(dataset1, dataset2)	#required to reset dataloader and still support tqdm modification
					else:
						loader = ANNpt_data.createDataLoaderTabular(dataset)	#required to reset dataloader and still support tqdm modification
				elif(useImageDataset):
					loader = ANNpt_data.createDataLoaderImage(dataset)	#required to reset dataloader and still support tqdm modification
				loop = tqdm(loader, leave=True)
				for batchIndex, batch in enumerate(loop):

					if(trainOrTest):
						loss, accuracy = trainBatch(batchIndex, batch, model, optim, l)
					else:
						loss, accuracy = testBatch(batchIndex, batch, model, l)

					if(printAccuracyRunningAverage):
						(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))

					loop.set_description(f'Epoch {epoch}')
					loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		saveModel(model)
					
def trainBatch(batchIndex, batch, model, optim, l=None):
	if(not trainLocal):
		optim.zero_grad()
	loss, accuracy = propagate(True, batchIndex, batch, model, optim, l)
	if(not trainLocal):
		loss.backward()
		optim.step()
	
	if(useSignedWeights):
		if(usePositiveWeightsClampModel):
			ANNpt_linearSublayers.weightsSetPositiveModel(model)

	if(batchIndex % modelSaveNumberOfBatches == 0):
		saveModel(model)
	loss = loss.item()
			
	return loss, accuracy
			
def testBatch(batchIndex, batch, model, l=None):

	loss, accuracy = propagate(False, batchIndex, batch, model, l)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy

def saveModel(model):
	torch.save(model, modelPathNameFull)

def loadModel():
	print("loading existing model")
	model = torch.load(modelPathNameFull)
	return model
		
def propagate(trainOrTest, batchIndex, batch, model, optim=None, l=None):
	(x, y) = batch
	y = y.long()
	x = x.to(device)
	y = y.to(device)
	if(debugDataNormalisation):
		print("x = ", x)
		print("y = ", y)
		
	loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy
				
if(__name__ == '__main__'):
	main()







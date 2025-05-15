"""ANNpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

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
pip install torchvision
pip install torchsummary

# Usage:
source activate pytorchsenv
python ANNpt_main.py

# Description:
ANNpt main - custom artificial neural network trained on tabular/image data

"""

import torch
from tqdm.auto import tqdm
from torch import optim
from torch.optim.lr_scheduler import StepLR, LambdaLR, SequentialLR



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
elif(useAlgorithmEISANI):
	import EISANIpt_EISANI as ANNpt_algorithm
elif(useAlgorithmAEANN):
	import AEANNpt_AEANN as ANNpt_algorithm
elif(useAlgorithmFFANN):
	import AEANNpt_FFANN as ANNpt_algorithm
	
if(useSignedWeights):
	import ANNpt_linearSublayers
import ANNpt_data

#https://huggingface.co/docs/datasets/tabular_load

def main():
	dataset = ANNpt_data.loadDataset()
	if(stateTrainDataset):
		model = ANNpt_algorithm.createModel(dataset[datasetSplitNameTrain])	#dataset[datasetSplitNameTest] not possible as test does not contain all classes
		processDataset(True, dataset[datasetSplitNameTrain], model)
	if(stateTestDataset):
		model = loadModel()
		processDataset(False, dataset[datasetSplitNameTest], model)

def createOptimizer():
	if(optimiserAdam):
		optim = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
	else:
		optim = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
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

def createScheduler(model, optim):
	def make_scheduler(opt):
		if warmupEpochs > 0:
			# 1) warm‑up λ(epoch): epoch/warmupEpochs clipped at 1.0
			warm = LambdaLR(opt, lr_lambda=lambda e: min(1.0, float(e + 1) / warmupEpochs))
			# 2) then step down by gamma every Stepsize
			step = StepLR(opt, step_size=learningRateSchedulerStepsize, gamma=learningRateSchedulerGamma)
			return SequentialLR(opt, schedulers=[warm, step], milestones=[warmupEpochs])
		else:
			return StepLR(opt, step_size=learningRateSchedulerStepsize, gamma=learningRateSchedulerGamma)

	if trainLocal:
		return [make_scheduler(o) for o in optim]
	else:
		return [make_scheduler(optim)]

def processDataset(trainOrTest, dataset, model):
	if(trainOrTest):
		if(useAlgorithmEISANI and trainLocal):
			optim = []
		elif(useAlgorithmEIANN and trainLocal):
			optim = []
			optim += [createOptimiser(model)]
			optim += [createOptimiser(model)]
		else:
			optim = createOptimiser(model)
		if(useLearningRateScheduler):
			schedulers = createScheduler(model, optim)
		model.to(device)
		model.train()	
		numberOfEpochs = trainNumberOfEpochs
	else:
		model.to(device)
		model.eval()
		numberOfEpochs = 1
		totalAccuracy = 0.0
		totalAccuracyCount = 0

	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)
		
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
						loss, accuracy = trainBatch(batchIndex, batch, model, optim, l, fieldTypeList)
					else:
						loss, accuracy = testBatch(batchIndex, batch, model, l, fieldTypeList)

					if(l == maxLayer-1):
						if(not trainOrTest):
							totalAccuracy = totalAccuracy + accuracy
							totalAccuracyCount += 1
							
					if(printAccuracyRunningAverage):
						(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
					
					loop.set_description(f'Epoch {epoch}')
					loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)
		
			if(not trainOrTest):
				averageAccuracy = totalAccuracy/totalAccuracyCount
				print("test averageAccuracy = ", averageAccuracy)

		if(trainOrTest):
			if(useLearningRateScheduler):
				for sch in schedulers:
					sch.step()
		
		if(useAlgorithmEISANI):
			model.executePostTrainPrune(trainOrTest)

		saveModel(model)
					
def trainBatch(batchIndex, batch, model, optim, l=None, fieldTypeList=None):
	if(not trainLocal):
		optim.zero_grad()
	loss, accuracy = propagate(True, batchIndex, batch, model, optim, l, fieldTypeList)
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
			
def testBatch(batchIndex, batch, model, l=None, fieldTypeList=None):
		
	loss, accuracy = propagate(False, batchIndex, batch, model, None, l, fieldTypeList)

	loss = loss.item()
	#loss = loss.detach().cpu().numpy()
	
	return loss, accuracy

def saveModel(model):
	torch.save(model, modelPathNameFull)

def loadModel():
	print("loading existing model")
	model = torch.load(modelPathNameFull, weights_only=False)
	return model
		
def propagate(trainOrTest, batchIndex, batch, model, optim=None, l=None, fieldTypeList=None):
	(x, y) = batch
	y = y.long()
	x = x.to(device)
	y = y.to(device)
	if(debugDataNormalisation):
		print("x = ", x)
		print("y = ", y)
	
	if(supportFieldTypeList):
		loss, accuracy = model(trainOrTest, x, y, optim, l, fieldTypeList)
	else:
		loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy
				
if(__name__ == '__main__'):
	main()







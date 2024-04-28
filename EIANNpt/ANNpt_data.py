"""ANNpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt data 

"""


import torch as pt
from datasets import load_dataset
from ANNpt_globalDefs import *
import numpy as np
import random
if(useImageDataset):
	import torchvision
	import torchvision.transforms as transforms

def loadDataset():
	if(useTabularDataset):
		return loadDatasetTabular()
	elif(useImageDataset):
		return loadDatasetImage()
		
def loadDatasetTabular():
	if(datasetLocalFile):
		trainFileNameFull = dataPathName + '/' + trainFileName
		testFileNameFull = dataPathName + '/' +  testFileName
		dataset = load_dataset('csv', data_files={"train":trainFileNameFull, "test":testFileNameFull})
	else:
		dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName, "test":testFileName})

	if(datasetNormalise):
		dataset['train'] = normaliseDataset(dataset['train'])
		dataset['test'] = normaliseDataset(dataset['test'])
	if(datasetRepeat):
		dataset['train'] = repeatDataset(dataset['train'])
		dataset['test'] = repeatDataset(dataset['test'])
	if(datasetShuffle):
		dataset = shuffleDataset(dataset)
	if(datasetOrderByClass):
		dataset = orderDatasetByClass(dataset)
		
	if(datasetNormaliseClassValues):
		dataset['train'] = normaliseClassValues(dataset['train'])
		dataset['test'] = normaliseClassValues(dataset['test'])
	
	return dataset

def normaliseDataset(dataset):
	datasetSize = getDatasetSize(dataset)
	for featureName in list(dataset.features):
		if(featureName != classFieldName):
			featureDataList = []
			for i in range(datasetSize):
				row = dataset[i]
				featureCell = row[featureName]
				featureDataList.append(featureCell)
			featureData = np.array(featureDataList)
			if(datasetNormaliseMinMax):
				featureMin = np.amin(featureData)
				featureMax = np.amax(featureData)
				featureData = (featureData - featureMin) / (featureMax - featureMin) #featureData/featureMax
			elif(datasetNormaliseStdAvg):
				featureMean = np.mean(featureData)
				featureStd = np.std(featureData)
				featureData = featureData-featureMean
				featureData = featureData/featureStd
			featureDataList = featureData.tolist()
			dataset = dataset.remove_columns(featureName)
			dataset = dataset.add_column(featureName, featureDataList)
	return dataset
	
def repeatDataset(dataset):
	datasetSize = getDatasetSize(dataset)
	repeatIndices = list(range(datasetSize))
	repeatIndices = repeatIndices*datasetRepeatSize
	dataset = dataset.select(repeatIndices)
	return dataset

def shuffleDataset(dataset):
	datasetSize = getDatasetSize(dataset)
	dataset = dataset.shuffle()
	return dataset
	
def orderDatasetByClass(dataset):
	dataset = dataset.sort(classFieldName)
	return dataset

def normaliseClassValues(dataset):
	classIndex = 0
	classIndexDict = {}
	classFieldNew = []
	datasetSize = getDatasetSize(dataset)
	numberOfClasses = 0
	for i in range(datasetSize):
		row = dataset[i]
		
		targetString = row[classFieldName]
		if(targetString in classIndexDict):
			target = classIndexDict[targetString]
			classFieldNew.append(target)
		else:
			target = classIndex
			classFieldNew.append(target)
			classIndexDict[targetString] = classIndex
			classIndex = classIndex + 1
		
	dataset = dataset.remove_columns(classFieldName)
	dataset = dataset.add_column(classFieldName, classFieldNew)

	return dataset
				
def countNumberClasses(dataset, printSize=False):
	numberOfClassSamples = {}
	datasetSize = getDatasetSize(dataset)
	numberOfClasses = 0
	for i in range(datasetSize):
		row = dataset[i]
		
		target = int(row[classFieldName])
			
		if(target in numberOfClassSamples):
			numberOfClassSamples[target] = numberOfClassSamples[target] + 1
		else:
			numberOfClassSamples[target] = 0
			
		#print("target = ", target)
		if(target > numberOfClasses):
			numberOfClasses = target
	numberOfClasses = numberOfClasses+1
	
	if(printSize):
		print("numberOfClasses = ", numberOfClasses)
	return numberOfClasses, numberOfClassSamples

def countNumberFeatures(dataset, printSize=False):
	numberOfFeatures = len(dataset.features)-1	#-1 to ignore class targets
	if(printSize):
		print("numberOfFeatures = ", numberOfFeatures)
	return numberOfFeatures
	
def getDatasetSize(dataset, printSize=False):
	datasetSize = dataset.num_rows
	if(printSize):
		print("datasetSize = ", datasetSize)
	return datasetSize
	
def createDataLoader(dataset):
	return createDataLoaderTabular(dataset)
	
def createDataLoaderTabular(dataset):
	dataLoaderDataset = DataloaderDatasetTabular(dataset)	
	maintainEvenBatchSizes = True
	if(dataloaderRepeatSampler):
		numberOfSamples = getDatasetSize(dataset)*dataloaderRepeatSize
		if(dataloaderRepeatSamplerCustom):
			sampler = CustomRandomSampler(dataset, shuffle=True, num_samples=numberOfSamples)
		else:
			sampler = pt.utils.data.RandomSampler(dataset, replacement=True, num_samples=numberOfSamples)
		loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, drop_last=dataloaderMaintainBatchSize, sampler=sampler)
	else:
		loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, shuffle=dataloaderShuffle, drop_last=dataloaderMaintainBatchSize)
	return loader
	
def createDataLoaderTabularPaired(dataset1, dataset2):
	dataLoaderDataset = DataloaderDatasetTabularPaired(dataset1, dataset2)	
	if(dataloaderRepeatSampler):
		numberOfSamples = getDatasetSize(dataset1)*dataloaderRepeatSize
		if(dataloaderRepeatSamplerCustom):
			sampler = CustomRandomSampler(dataset1, shuffle=True, num_samples=numberOfSamples)
		else:
			sampler = pt.utils.data.RandomSampler(dataset1, replacement=True, num_samples=numberOfSamples)
		loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, drop_last=dataloaderMaintainBatchSize, sampler=sampler)
	else:
		loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, shuffle=dataloaderShuffle, drop_last=dataloaderMaintainBatchSize)
	return loader

class DataloaderDatasetTabular(pt.utils.data.Dataset):
	def __init__(self, dataset):
		self.datasetSize = getDatasetSize(dataset)
		self.dataset = dataset
		self.datasetIterator = iter(dataset)
			
	def __len__(self):
		return self.datasetSize

	def __getitem__(self, i):
		if(dataloaderRepeatSampler):
			try:
				document = next(self.datasetIterator)
			except StopIteration:
				self.datasetIterator = iter(self.dataset)
				document = next(self.datasetIterator)
		else:
			document = next(self.datasetIterator)
		documentList = list(document.values())
		if(datasetReplaceNoneValues):
			documentList = [x if x is not None else 0 for x in documentList]
		#print("documentList = ", documentList)
		x = documentList[0:-1]
		y = documentList[-1]
		x = pt.Tensor(x).float()
		batchSample = (x, y)
		return batchSample
		
class DataloaderDatasetTabularPaired(pt.utils.data.Dataset):
	def __init__(self, dataset1, dataset2):
		self.datasetSize = getDatasetSize(dataset1)
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		self.datasetIterator1 = iter(dataset1)
		self.datasetIterator2 = iter(dataset2)

	def __len__(self):
		return self.datasetSize

	def __getitem__(self, i):
		if(dataloaderRepeatSampler):
			try:
				document1 = next(self.datasetIterator1)
				document2 = next(self.datasetIterator2)
			except StopIteration:
				self.datasetIterator1 = iter(self.dataset1)
				self.datasetIterator2 = iter(self.dataset2)
				document1 = next(self.datasetIterator1)
				document2 = next(self.datasetIterator2)
		else:
			document1 = next(self.datasetIterator1)
			document2 = next(self.datasetIterator2)
		documentList1 = list(document1.values())
		documentList2 = list(document2.values())
		if(datasetReplaceNoneValues):
			documentList1 = [x if x is not None else 0 for x in documentList1]
			documentList2 = [x if x is not None else 0 for x in documentList2]
		#print("documentList = ", documentList)
		x1 = documentList1[0:-1]
		x2 = documentList2[0:-1]
		x1 = pt.Tensor(x1).float()
		x2 = pt.Tensor(x2).float()
		x1 = pt.unsqueeze(x1, dim=0)
		x2 = pt.unsqueeze(x2, dim=0)
		x = pt.concat([x1, x2], dim=0)
		y1 = documentList1[-1]
		y2 = documentList2[-1]
		#print("y1 = ", y1, ", y2 = ", y2)	#verify they are equal
		y = y1
		batchSample = (x, y)
		return batchSample
		
class CustomRandomSampler(pt.utils.data.Sampler):
	def __init__(self, dataset, shuffle, num_samples):
		self.dataset = dataset
		self.shuffle = shuffle
		self.num_samples = num_samples

	def __iter__(self):
		order = list(range((getDatasetSize(self.dataset))))
		idx = 0
		sampleIndex = 0
		while sampleIndex < self.num_samples:
			#print("idx = ", idx)
			#print("order[idx] = ", order[idx])
			yield order[idx]
			idx += 1
			if idx == len(order):
				if self.shuffle:
					random.shuffle(order)
				idx = 0
			sampleIndex += 1

def loadDatasetImage():
	# Load the CIFAR-10 dataset and define preprocessing transformations
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dataset = {}
	dataset['train'] = torchvision.datasets.CIFAR10(root=dataPathName, train=True, download=True, transform=transform)
	dataset['test'] = torchvision.datasets.CIFAR10(root=dataPathName, train=False, download=True, transform=transform)
	return dataset

def createDataLoaderImage(dataset):
	loader = pt.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
	return loader
	

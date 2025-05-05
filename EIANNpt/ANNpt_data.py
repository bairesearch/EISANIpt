"""ANNpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

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
from datasets import load_dataset, Value
from ANNpt_globalDefs import *
import numpy as np
import random
if(useImageDataset):
	import torchvision
	import torchvision.transforms as transforms
import pyarrow as pa

if(disableDatasetCache):
	import datasets
	from datasets import disable_caching
	disable_caching()
	datasets.config.IN_MEMORY_MAX_SIZE = 32 * 10**9	#in bytes

debugSaveRawDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveSplitDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveNormalisedDatasetToCSV = False	#output dataset to csv file for manual checks

def loadDataset():
	if(useTabularDataset):
		return loadDatasetTabular()
	elif(useImageDataset):
		return loadDatasetImage()

def saveDatasetToCSV(dataset):
	for split in dataset.keys():  # Usually 'train', 'test', etc.
		output_file = f"{datasetName}_{split}.csv"
		dataset[split].to_csv(output_file)
		print(f"Saved {split} split to {output_file}")
				
def loadDatasetTabular():
	if(datasetLocalFile):
		trainFileNameFull = dataPathName + '/' + trainFileName
		if(datasetHasTestSplit):
			testFileNameFull = dataPathName + '/' +  testFileName
			dataset = load_dataset('csv', data_files={"train":trainFileNameFull, "test":testFileNameFull})
		else:
			dataset = load_dataset('csv', data_files={"train":trainFileNameFull})
	else:
		if(datasetSpecifyDataFiles):
			if(datasetHasTestSplit):
				dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName, "test":testFileName})
			else:
				dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName})
		elif(datasetHasSubsetType):
			dataset = load_dataset(datasetNameFull, datasetSubsetName)
		else:
			dataset = load_dataset(datasetNameFull)

	if(debugSaveRawDatasetToCSV):
		saveDatasetToCSV(dataset)
		
	if(datasetConvertFeatureValues):
		dataset[datasetSplitNameTrain] = convertFeatureValues(dataset[datasetSplitNameTrain])
		if(datasetHasTestSplit):
			dataset[datasetSplitNameTest] = convertFeatureValues(dataset[datasetSplitNameTest])
	if(datasetConvertClassValues):
		dataset[datasetSplitNameTrain] = convertClassValues(dataset[datasetSplitNameTrain])
		if(datasetHasTestSplit):
			dataset[datasetSplitNameTest] = convertClassValues(dataset[datasetSplitNameTest])
	else:
		if(datasetConvertClassTargetColumnFloatToInt):
			dataset[datasetSplitNameTrain] = convertClassTargetColumnFloatToInt(dataset[datasetSplitNameTrain])
			if(datasetHasTestSplit):
				dataset[datasetSplitNameTest] = convertClassTargetColumnFloatToInt(dataset[datasetSplitNameTest])
							
	if(not datasetHasTestSplit):
		dataset[datasetSplitNameTrain] = shuffleDataset(dataset[datasetSplitNameTrain])
		dataset = dataset[datasetSplitNameTrain].train_test_split(test_size=datasetTestSplitSize)

	if(debugSaveSplitDatasetToCSV):
		saveDatasetToCSV(dataset)
		
	if(datasetNormalise):
		dataset[datasetSplitNameTrain] = normaliseDataset(dataset[datasetSplitNameTrain])
		dataset[datasetSplitNameTest] = normaliseDataset(dataset[datasetSplitNameTest])
	if(datasetRepeat):
		dataset[datasetSplitNameTrain] = repeatDataset(dataset[datasetSplitNameTrain])
		dataset[datasetSplitNameTest] = repeatDataset(dataset[datasetSplitNameTest])
	if(datasetShuffle):
		dataset[datasetSplitNameTrain] = shuffleDataset(dataset[datasetSplitNameTrain])
		dataset[datasetSplitNameTest] = shuffleDataset(dataset[datasetSplitNameTest])
	if(datasetOrderByClass):
		dataset[datasetSplitNameTrain] = orderDatasetByClass(dataset[datasetSplitNameTrain])
		dataset[datasetSplitNameTest] = orderDatasetByClass(dataset[datasetSplitNameTest])
		#dataset = orderDatasetByClass(dataset)
	
	dataset[datasetSplitNameTrain] = repositionClassFieldToLastColumn(dataset[datasetSplitNameTrain])
	dataset[datasetSplitNameTest] = repositionClassFieldToLastColumn(dataset[datasetSplitNameTest])

	if(debugSaveNormalisedDatasetToCSV):
		saveDatasetToCSV(dataset)
		
	return dataset

def repositionClassFieldToLastColumn(dataset):
	classDataList = dataset[classFieldName]
	dataset = dataset.remove_columns(classFieldName)
	dataset = dataset.add_column(classFieldName, classDataList)
	return dataset
			
def convertClassTargetColumnFloatToInt(dataset):
	classDataList = dataset[classFieldName]
	classDataList = [int(value) for value in classDataList]
	dataset = dataset.remove_columns(classFieldName)
	dataset = dataset.add_column(classFieldName, classDataList)
	return dataset


def normaliseDataset(dataset):
	print("normaliseDataset:  dataset.num_rows = ",  dataset.num_rows, ", len(dataset.features) = ", len(dataset.features))
	datasetSize = getDatasetSize(dataset)
	for featureIndex, featureName in enumerate(list(dataset.features)):
		#print("featureIndex = ", featureIndex)
		if(featureName != classFieldName):
			if(datasetCorrectMissingValues):
				featureDataList = []
				for i in range(datasetSize):
					row = dataset[i]
					featureCell = row[featureName]
					if(featureCell == None):
						featureCell = 0
					featureDataList.append(featureCell)
			else:
				featureDataList = dataset[featureName]
			featureData = np.array(featureDataList)
			#print("featureData = ", featureData)
			if(datasetNormaliseMinMax):
				featureMin = np.amin(featureData)
				featureMax = np.amax(featureData)
				#if(featureMax - featureMin == 0):
				#	print("warning: (featureMax - featureMin == 0)")
				featureData = (featureData - featureMin) / (featureMax - featureMin + 1e-8) #featureData/featureMax
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

def convertFeatureValues(dataset):
	print("convertFeatureValues:  dataset.num_rows = ",  dataset.num_rows, ", len(dataset.features) = ", len(dataset.features))
	for fieldName, fieldType in dataset.features.items():
		#print("convertFeatureValues: fieldName = ", fieldName)
		if fieldType.dtype == 'string':
			dataset = convertCategoricalFieldValues(dataset, fieldName)
		elif fieldType.dtype == 'bool':
			dataset = dataset.cast_column(fieldName, Value('float32'))
	return dataset

def bool_to_float(example):
    example[fieldName] = float(example[fieldName])
    return example

	
def convertClassValues(dataset):
	return convertCategoricalFieldValues(dataset, classFieldName, dataType=int)

def convertCategoricalFieldValues(dataset, fieldName, dataType=float):
	if(not (dataType==float or dataType==int)):
		printe("convertCategoricalFieldValues error: not (dataType==float or dataType==int)")
		
	#print("convertCategoricalFieldValues: fieldName = ", fieldName)
	fieldIndex = 0
	fieldIndexDict = {}
	fieldNew = []
	datasetSize = getDatasetSize(dataset)
	#print("datasetSize = ", datasetSize)
	numberOfClasses = 0
	for i in range(datasetSize):
		row = dataset[i]
		#print("i = ", i)
		
		targetString = row[fieldName]
		if(targetString in fieldIndexDict):
			target = fieldIndexDict[targetString]
			if(dataType==float):
				target = float(target)
			fieldNew.append(target)
		else:
			target = fieldIndex
			if(dataType==float):
				target = float(target)
			fieldNew.append(target)
			fieldIndexDict[targetString] = fieldIndex
			fieldIndex = fieldIndex + 1
		
	dataset = dataset.remove_columns(fieldName)
	dataset = dataset.add_column(fieldName, fieldNew)

	return dataset

def normaliseBooleanFieldValues(dataset, fieldName, dataType=float):
	if(not (dataType==float or dataType==int)):
		printe("normaliseBooleanFieldValues error: not (dataType==float or dataType==int)")

	fieldNew = []
	datasetSize = getDatasetSize(dataset)
	for i in range(datasetSize):
		row = dataset[i]
		print("i = ", i)
		targetBool = row[fieldName]
		if(targetBool == True):
			target = 1
		elif(targetBool == False):
			target = 0
		if(dataType==float):
			target = float(target)
		fieldNew.append(target)
		
	dataset = dataset.remove_columns(fieldName)
	dataset = dataset.add_column(fieldName, fieldNew)

	return dataset
	
def countNumberClasses(dataset, printSize=False):
	numberOfClassSamples = {}
	datasetSize = getDatasetSize(dataset)
	numberOfClasses = 0
	for i in range(datasetSize):
		if useImageDataset:
			# For PyTorch datasets (e.g., CIFAR10), the label is the second element of the dataset tuple
			_, target = dataset[i]
		elif useTabularDataset:
			# For Hugging Face datasets, the label is accessed via the classFieldName
			row = dataset[i]
			target = int(row[classFieldName])
		else:
			raise AttributeError("Unsupported dataset type: Unable to count classes.")
			
		if(target in numberOfClassSamples):
			numberOfClassSamples[target] = numberOfClassSamples[target] + 1
		else:
			numberOfClassSamples[target] = 0
			
		#print("target = ", target)
		if(target > numberOfClasses):
			numberOfClasses = target
	numberOfClasses = numberOfClasses+1
	
	#if(printSize):
	#print("numberOfClasses = ", numberOfClasses)
	return numberOfClasses, numberOfClassSamples

def countNumberFeatures(dataset, printSize=False):
	if useImageDataset:
		# For image datasets, infer the number of features from the image shape
		sample_image, _ = dataset[0]  # Get the first sample (image, label)
		numberOfFeatures = sample_image.shape[0]*sample_image.shape[1]*sample_image.shape[2]
	elif useTabularDataset:
		# For tabular datasets, use the features attribute
		numberOfFeatures = len(dataset.features) - 1  # -1 to ignore class targets
	else:
		raise AttributeError("Unsupported dataset type: Unable to determine the number of features.")
	
	if(printSize):
		print("numberOfFeatures = ", numberOfFeatures)
	return numberOfFeatures
	
def getDatasetSize(dataset, printSize=False):
	if useImageDataset:
		# Check if the dataset is a PyTorch dataset (e.g., CIFAR10)
		datasetSize = len(dataset)
	elif useTabularDataset:
		# Otherwise, assume it's a Hugging Face dataset
		datasetSize = dataset.num_rows
	else:
		raise AttributeError("Unsupported dataset type: Unable to determine the number of features.")
	
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
			#index = i % self.datasetSize
			#document = self.dataset[index]
			try:
				document = next(self.datasetIterator)	#does not support dataloaderShuffle
			except StopIteration:
				self.datasetIterator = iter(self.dataset)
				document = next(self.datasetIterator)	#does not support dataloaderShuffle
		else:
			 document = self.dataset[i]	
			 #document = next(self.datasetIterator) #does not support dataloaderShuffle
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
			#index = i % self.datasetSize
			#document1 = self.dataset1[index]
			#document2 = self.dataset2[index]
			try:
				document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
				document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
			except StopIteration:
				self.datasetIterator1 = iter(self.dataset1)
				self.datasetIterator2 = iter(self.dataset2)
				document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
				document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
		else:
			document1 = self.dataset1[i]
			document2 = self.dataset2[i]	
			#document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
			#document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
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
	# Load the CIFAR-10 dataset with optional augmentation
	if imageDatasetAugment:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
			transforms.Lambda(lambda img: cutout(img))
		])
	else:
		train_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])
	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = {}
	dataset[datasetSplitNameTrain] = torchvision.datasets.CIFAR10(root=dataPathName, train=True, download=True, transform=train_transform)
	dataset[datasetSplitNameTest] = torchvision.datasets.CIFAR10(root=dataPathName, train=False, download=True, transform=test_transform)
	return dataset

def createDataLoaderImage(dataset):
	loader = pt.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
	return loader

def cutout(img, n_holes=1, length=16):
	"""Apply cutout augmentation to a tensor image."""
	h, w = img.size(1), img.size(2)
	mask = np.ones((h, w), np.float32)
	for _ in range(n_holes):
		y = random.randrange(h)
		x = random.randrange(w)
		y1 = max(0, y - length//2)
		y2 = min(h, y + length//2)
		x1 = max(0, x - length//2)
		x2 = min(w, x + length//2)
		mask[y1:y2, x1:x2] = 0.
	mask = pt.from_numpy(mask)
	mask = mask.expand_as(img)
	return img * mask

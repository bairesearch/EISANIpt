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
from datasets import load_dataset, Value, Features
from ANNpt_globalDefs import *
import ANNpt_globalDefs
import numpy as np
import random
if(useTabularDataset):
	pass
elif(useImageDataset):
	import torchvision
	import torchvision.transforms as transforms
elif(useNLPDataset):
	from transformers import AutoTokenizer, DataCollatorWithPadding
	from torch.utils.data import IterableDataset as TorchIterableDataset, DataLoader
	from datasets import Dataset, IterableDataset as HFDIterable, get_dataset_config_info, DatasetDict
	bert_pad_id = None
	bert_tokenizer = None
	import string
import pyarrow as pa
import pyarrow.compute as pc

if(disableDatasetCache):
	import datasets
	from datasets import disable_caching
	disable_caching()
	datasets.config.IN_MEMORY_MAX_SIZE = 32 * 10**9	#in bytes

debugSaveRawDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveSplitDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveNormalisedDatasetToCSV = False	#output dataset to csv file for manual checks

_cachedFeatureTypeList = None
_cachedFeatureTypeMap = None
_cachedClassFieldType = None


def _extract_feature_dtype(feature):
	"""Best-effort extraction of a Hugging Face feature dtype."""
	dtype = getattr(feature, 'dtype', None)
	if dtype is None and hasattr(feature, 'feature'):
		inner_feature = getattr(feature, 'feature')
		dtype = getattr(inner_feature, 'dtype', None)
	if dtype is None and hasattr(feature, 'value_type'):
		inner_feature = getattr(feature, 'value_type')
		dtype = getattr(inner_feature, 'dtype', None)
	return dtype

def _get_arrow_table(dataset_split):
    # Works across HF 2.x releases
    table = getattr(dataset_split, "data", None)
    if table is None:
        table = getattr(dataset_split, "_data", None)
    if table is None:
        table = dataset_split.to_table()  # last resort (avoid in tight loops)
    return table

def _is_binary_arrow_column(column):
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if len(column) == 0 or column.null_count == len(column):
        return False
    non_null = pc.drop_null(column)
    if len(non_null) == 0:
        return False
    col_type = non_null.type
    if pa.types.is_integer(col_type):
        limits = pc.min_max(non_null)
        return limits["min"].as_py() >= 0 and limits["max"].as_py() <= 1
    if pa.types.is_floating(col_type):
        allowed = pa.array([0.0, 1.0], type=col_type)
        mask = pc.is_in(non_null, value_set=allowed)
        return bool(pc.all(mask).as_py())
    return False

def _cache_tabular_field_types(dataset_dict):
    global _cachedFeatureTypeList, _cachedFeatureTypeMap, _cachedClassFieldType
    _cachedFeatureTypeList = _cachedFeatureTypeMap = None
    _cachedClassFieldType = None

    if dataset_dict is None:
        printe("_cache_tabular_field_types error: dataset_dict is None")
        return

    reference_split = dataset_dict.get(datasetSplitNameTrain)
    if reference_split is None:
        printe("_cache_tabular_field_types error: reference_split is None")
        return

    if 'features' in reference_split.column_names and len(reference_split.column_names) == 2:
        printe("_cache_tabular_field_types error: 'features' already consolidated")
        return

    feature_names = [
        name for name in reference_split.column_names
        if name not in (classFieldName, 'features')
    ]
    if not feature_names:
        printe("_cache_tabular_field_types error: not feature_names")
        return

    arrow_table = _get_arrow_table(reference_split)

    _cachedFeatureTypeList = []
    _cachedFeatureTypeMap = {}

    for feature_name in feature_names:
        feature_info = reference_split.features[feature_name]
        dtype = _extract_feature_dtype(feature_info)

        if dtype in {
            'float16', 'float32', 'float64',
            'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64'
        }:
            idx = arrow_table.schema.get_field_index(feature_name)
            if idx != -1 and _is_binary_arrow_column(arrow_table.column(idx)):
                dtype = 'bool'

        _cachedFeatureTypeList.append(dtype)
        _cachedFeatureTypeMap[feature_name] = dtype

    class_info = reference_split.features.get(classFieldName)
    if class_info is not None:
        _cachedClassFieldType = _extract_feature_dtype(class_info)

def loadDataset():
	if(useTabularDataset):
		return loadDatasetTabular()
	elif(useImageDataset):
		return loadDatasetImage()
	elif(useNLPDataset):
		return loadDatasetNLP()

def saveDatasetToCSV(dataset):
	for split in dataset.keys():  # Usually 'train', 'test', etc.
		output_file = f"{datasetName}_{split}.csv"
		dataset[split].to_csv(output_file)
		print(f"Saved {split} split to {output_file}")
				
	
def countNumberClasses(dataset, printSize=False):
	if useTabularDataset:
		classValues = dataset[classFieldName]
		#if(useImageDataset): classValues = [dataset[i][1] for i in range(getDatasetSize(dataset))]

		if isinstance(classValues, pt.Tensor):
			if classValues.numel() == 0:
				return 0, {}
			classArray = classValues.cpu().numpy()
		else:
			if len(classValues) == 0:
				return 0, {}
			classArray = np.array(classValues)
		if np.issubdtype(classArray.dtype, np.floating):
			classArray = classArray.astype(np.int64)
		elif np.issubdtype(classArray.dtype, np.bool_):
			classArray = classArray.astype(np.int64)

		uniqueClasses, classCounts = np.unique(classArray, return_counts=True)
		numberOfClasses = int(uniqueClasses.max()) + 1
		numberOfClassSamples = {int(classId): int(count) for classId, count in zip(uniqueClasses, classCounts)}
	elif(useImageDataset):
		numberOfClasses = ANNpt_globalDefs.numberOfClasses
		numberOfClassSamples = None	#not used
	elif(useNLPDataset):
		numberOfClasses = ANNpt_globalDefs.numberOfClasses
		numberOfClassSamples = None	#not used

	#if(printSize):
	#	print("numberOfClasses = ", numberOfClasses)

	return numberOfClasses, numberOfClassSamples

def countNumberFeatures(dataset, printSize=False):
	if useImageDataset:
		# For image datasets, infer the number of features from the image shape
		sample_image, _ = dataset[0]  # Get the first sample (image, label)
		numberOfFeatures = sample_image.shape[0]*sample_image.shape[1]*sample_image.shape[2]
	elif useTabularDataset:
		if('features' in dataset.column_names):
			sample_features = dataset[0]['features']
			if isinstance(sample_features, pt.Tensor):
				numberOfFeatures = sample_features.shape[-1]
			else:
				numberOfFeatures = len(sample_features)
		else:
			numberOfFeatures = len(dataset.features) - 1  # -1 to ignore class targets	#orig implementation
	elif useNLPDataset:
		if(useTokenEmbedding):
			numberOfFeatures = contextSizeMax * embeddingSize
		else:
			numberOfFeatures = contextSizeMax
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
	elif useNLPDataset:
		datasetSize = datasetSizeRecord
	else:
		raise AttributeError("Unsupported dataset type: Unable to determine the number of features.")
	
	if(printSize):
		print("datasetSize = ", datasetSize)
	return datasetSize

def createFieldTypeList(dataset):
	if(useTabularDataset):
		if _cachedFeatureTypeList is not None:
			fieldTypeList = list(_cachedFeatureTypeList)
			if _cachedClassFieldType is not None:
				fieldTypeList.append(_cachedClassFieldType)
		else:
			printe("createFieldTypeList requires _cachedFeatureTypeList")
	else:
		fieldTypeList = None
	return fieldTypeList

def createDataLoader(dataset):
	return createDataLoaderTabular(dataset)

if(useTabularDataset):

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

		if(debugCullDatasetSamples):
			dataset['train'] = dataset['train'].select(range(100))
			if(datasetHasTestSplit):
				dataset['test'] = dataset['test'].select(range(100))

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

		if(datasetName == 'topquark'):
			dataset = removeTopQuarkLeakageColumns(dataset)
								
		if(not datasetHasTestSplit):
			dataset[datasetSplitNameTrain] = shuffleDataset(dataset[datasetSplitNameTrain])
			dataset = dataset[datasetSplitNameTrain].train_test_split(test_size=datasetTestSplitSize)

		if(debugSaveSplitDatasetToCSV):
			saveDatasetToCSV(dataset)
			
		if(datasetEqualiseClassSamples):
			dataset[datasetSplitNameTrain] = equaliseClassSamples(dataset[datasetSplitNameTrain])
			if(datasetHasTestSplit and datasetEqualiseClassSamplesTest):
				dataset[datasetSplitNameTest] = equaliseClassSamples(dataset[datasetSplitNameTest])

		_cache_tabular_field_types(dataset)

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
		
		dataset = consolidateFeatureColumns(dataset)
		dataset = setDatasetTorchFormat(dataset)

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


	def removeTopQuarkLeakageColumns(datasetDict):
		leakageSubstrings = ["truth", "is_signal", "label", "target"]
		leakageExact = {"ttv"}

		for splitName in datasetDict.keys():
			split = datasetDict[splitName]
			columnsToRemove = []
			for columnName in split.column_names:
				if columnName == classFieldName:
					continue
				columnNameLower = columnName.lower()
				if columnNameLower in leakageExact:
					columnsToRemove.append(columnName)
					continue
				if any(substring in columnNameLower for substring in leakageSubstrings):
					columnsToRemove.append(columnName)
			if columnsToRemove:
				split = split.remove_columns(columnsToRemove)
				datasetDict[splitName] = split
		return datasetDict

	def equaliseClassSamples(dataset):
		#Equalises the number of samples across each class by repeating class samples as necessary.

		_, numberOfClassSamples = countNumberClasses(dataset) #dict: {class_label: count}
		if not numberOfClassSamples: # Handles empty or single-class datasets gracefully
			printe("No classes found in the dataset.")

		max_samples = 0
		if numberOfClassSamples: # Ensure there's at least one class to avoid error with max() on empty sequence
			max_samples = max(numberOfClassSamples.values())
		if max_samples == 0: # All classes have 0 samples or no classes
			printe("No samples found in any class.")

		class_specific_indices = {class_val: [] for class_val in numberOfClassSamples.keys()}
		for i in range(getDatasetSize(dataset)):
			row = dataset[i]
			# Assuming classFieldName holds the key to the class label and it's convertible to int
			# This part might need adjustment based on how class labels are stored/accessed
			try:
				target = int(row[classFieldName])
				if target in class_specific_indices:
					class_specific_indices[target].append(i)
			except (KeyError, ValueError) as e:
				print(f"Warning: Could not process class label for row {i}: {e}")
				continue # Skip rows where class label is problematic

		all_new_indices = []
		for class_val, count in numberOfClassSamples.items():
			current_indices = class_specific_indices.get(class_val, [])
			all_new_indices.extend(current_indices) # Add existing samples

			num_to_add = max_samples - count
			if num_to_add > 0 and current_indices: # Check if samples need to be added and if source samples exist
				repeated_indices = random.choices(current_indices, k=num_to_add)
				all_new_indices.extend(repeated_indices)

		if all_new_indices:
			# Shuffle to mix original and repeated samples, then select.
			# random.shuffle(all_new_indices) # Optional: shuffle before select if desired, otherwise select preserves order then shuffleDataset handles it later
			dataset = dataset.select(all_new_indices)

		return dataset

	def normaliseDataset(dataset):
		print("normaliseDataset:  dataset.num_rows = ",  dataset.num_rows, ", len(dataset.features) = ", len(dataset.features))

		featureNames = [featureName for featureName in dataset.column_names if featureName != classFieldName]
		if not featureNames:
			return dataset

		# Ensure feature columns are stored as floating point to avoid schema conflicts when returning float values
		if any(not isinstance(dataset.features[featureName], Value) or dataset.features[featureName].dtype not in ('float32', 'float64') for featureName in featureNames):
			featuresDict = {}
			for columnName, featureInfo in dataset.features.items():
				if columnName in featureNames:
					featuresDict[columnName] = Value('float64')
				else:
					featuresDict[columnName] = featureInfo
			dataset = dataset.cast(Features(featuresDict))

		normStats = {}
		for featureName in featureNames:
			'''
			TODO (codex):
			fieldType = dataset.features[featureName]
			if hasattr(fieldType, 'dtype') and fieldType.dtype == 'bool':
				continue #skip normalisation for boolean fields
			'''
			featureDataList = dataset[featureName]
			if(datasetCorrectMissingValues):
				featureDataList = [0 if value is None else value for value in featureDataList]
			featureData = np.array(featureDataList, dtype=np.float32)
			if(datasetNormaliseMinMax):
				featureMin = float(np.amin(featureData))
				featureMax = float(np.amax(featureData))
				featureRange = featureMax - featureMin
				if(featureRange == 0.0):
					normStats[featureName] = ("minmax", featureMin, 0.0)
				else:
					invRange = 1.0 / (featureRange + 1e-8)
					normStats[featureName] = ("minmax", featureMin, invRange)
			elif(datasetNormaliseStdAvg):
				featureMean = float(np.mean(featureData))
				featureStd = float(np.std(featureData))
				if(featureStd == 0.0):
					normStats[featureName] = ("std", featureMean, 0.0)
				else:
					invStd = 1.0 / (featureStd + 1e-8)
					normStats[featureName] = ("std", featureMean, invStd)

		if not normStats:
			return dataset

		def normaliseBatch(batch):
			for featureName, (normType, offset, scale) in normStats.items():
				values = batch[featureName]
				if(datasetCorrectMissingValues):
					values = [0 if value is None else value for value in values]
				valuesArray = np.asarray(values, dtype=np.float32)
				if(normType == "minmax"):
					if(scale == 0.0):
						valuesArray = valuesArray - offset
					else:
						valuesArray = (valuesArray - offset) * scale
				elif(normType == "std"):
					if(scale == 0.0):
						valuesArray = valuesArray - offset
					else:
						valuesArray = (valuesArray - offset) * scale
				batch[featureName] = valuesArray.tolist()
			return batch

		dataset = dataset.map(normaliseBatch, batched=True, batch_size=1024, desc="normaliseDataset", load_from_cache_file=True)
		return dataset

	def consolidateFeatureColumns(datasetDict):
		referenceSplit = datasetDict[datasetSplitNameTrain]
		if('features' in referenceSplit.column_names and len(referenceSplit.column_names) == 2):
			return datasetDict

		featureNames = [featureName for featureName in referenceSplit.column_names if featureName != classFieldName and featureName != 'features']
		if not featureNames:
			return datasetDict

		def packFeatures(batch):
			featureArrays = [np.asarray(batch[featureName], dtype=np.float32) for featureName in featureNames]
			featuresStacked = np.stack(featureArrays, axis=-1).astype(np.float32, copy=False)
			return {"features": featuresStacked}

		for splitName in datasetDict.keys():
			datasetDict[splitName] = datasetDict[splitName].map(
				packFeatures,
				batched=True,
				batch_size=1024,
				desc=f"packFeatures:{splitName}",
				remove_columns=featureNames
			)
		return datasetDict

	def setDatasetTorchFormat(datasetDict):
		for splitName in datasetDict.keys():
			datasetDict[splitName] = datasetDict[splitName].with_format(
				type="torch",
				columns=['features', classFieldName],
				output_all_columns=False
			)
		return datasetDict

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
				dataset = convertCategoricalFieldValues(dataset, fieldName, dataType=float)
			#elif fieldType.dtype == 'bool':
			#	dataset = dataset.cast_column(fieldName, Value('float32'))
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
			targetString = row[fieldName]
			if(targetString not in fieldIndexDict):
				fieldIndexDict[targetString] = fieldIndex
				fieldIndex = fieldIndex + 1		

		booleanCategoryDetected = False
		if(fieldIndex == 2):
			booleanCategoryDetected = True

		for i in range(datasetSize):
			row = dataset[i]
			targetString = row[fieldName]
			target = fieldIndexDict[targetString]
			if(dataType==int):	#always store class target as int (never bool)
				target = int(target)	#keep as int (redundant)
			elif(booleanCategoryDetected):
				target = bool(target)
			elif(dataType==float):
				target = float(target)
			fieldNew.append(target)

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

	def createDataLoaderTabular(dataset):
		dataLoaderDataset = DataloaderDatasetTabular(dataset)
		maintainEvenBatchSizes = True
		dataLoaderKwargs = {
			"batch_size": batchSize,
			"drop_last": dataloaderMaintainBatchSize,
			"num_workers": dataloaderNumWorkers,
			"pin_memory": dataloaderPinMemory,
		}
		if(dataloaderNumWorkers > 0):
			dataLoaderKwargs["persistent_workers"] = True

		if(dataloaderRepeatSampler):
			numberOfSamples = getDatasetSize(dataset)*dataloaderRepeatSize
			if(dataloaderRepeatSamplerCustom):
				sampler = CustomRandomSampler(dataset, shuffle=True, num_samples=numberOfSamples)
			else:
				sampler = pt.utils.data.RandomSampler(dataset, replacement=True, num_samples=numberOfSamples)
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, sampler=sampler, **dataLoaderKwargs)
		else:
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, shuffle=dataloaderShuffle, **dataLoaderKwargs)
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
				
		def __len__(self):
			return self.datasetSize

		def __getitem__(self, i):
			document = self.dataset[i]
			features = document['features']
			target = document[classFieldName]

			if isinstance(features, pt.Tensor):
				x = features.float()
			elif isinstance(features, np.ndarray):
				x = pt.from_numpy(features).float()
			else:
				x = pt.tensor(features, dtype=pt.float32)

			if isinstance(target, pt.Tensor):
				y = target.long()
			else:
				y = pt.tensor(int(target), dtype=pt.long)

			return x, y
			
	class DataloaderDatasetTabularPaired(pt.utils.data.Dataset):
		def __init__(self, dataset1, dataset2):
			self.datasetSize = getDatasetSize(dataset1)
			self.dataset1 = dataset1
			self.dataset2 = dataset2

		def __len__(self):
			return self.datasetSize

		def __getitem__(self, i):
			document1 = self.dataset1[i]
			document2 = self.dataset2[i]
			x1 = document1['features']
			x2 = document2['features']
			if isinstance(x1, pt.Tensor):
				x1 = x1.float()
			else:
				x1 = pt.tensor(x1, dtype=pt.float32)
			if isinstance(x2, pt.Tensor):
				x2 = x2.float()
			else:
				x2 = pt.tensor(x2, dtype=pt.float32)
			x1 = pt.unsqueeze(x1, dim=0)
			x2 = pt.unsqueeze(x2, dim=0)
			x = pt.concat([x1, x2], dim=0)
			y1 = document1[classFieldName]
			y2 = document2[classFieldName]
			#print("y1 = ", y1, ", y2 = ", y2)	#verify they are equal
			if isinstance(y1, pt.Tensor):
				y = y1.long()
			else:
				y = pt.tensor(int(y1), dtype=pt.long)
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

elif(useImageDataset):

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
		if(datasetName=="CIFAR10"):
			dataset[datasetSplitNameTrain] = torchvision.datasets.CIFAR10(root=dataPathName, train=True, download=True, transform=train_transform)
			dataset[datasetSplitNameTest] = torchvision.datasets.CIFAR10(root=dataPathName, train=False, download=True, transform=test_transform)
		else:
			printe("loadDatasetImage currently requires datasetName==CIFAR10")		
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

elif(useNLPDataset):

	def loadDatasetNLP():
		info = get_dataset_config_info(datasetName, datasetCfg)  # tiny JSON download
		datasetSize = info.splits["train"].num_examples
		base_stream = load_dataset(datasetName, datasetCfg, split="train", streaming=True, trust_remote_code=True)
		
		if(stateTestDataset):
			assert datasetSizeSubset, "loadDatasetNLP error: if stateTestDataset, datasetSizeSubset is required"
			if(not stateTrainDataset):
				print("loadDatasetNLP warning: stateTestDataset and !stateTrainDataset: assume train rows already streamed and cached (else will take long time to download train data before can start streaming test data)")
			#train_rows = int(datasetSize*(1-datasetTestSplitSize))
			#eval_rows = int(datasetSize*datasetTestSplitSize)
			train_rows = datasetTrainRows
			eval_rows = datasetTestRows
			train_stream = base_stream.take(train_rows)
			test_stream = base_stream.skip(train_rows).take(eval_rows)
			datasetSize = train_rows
		else:
			if(datasetSizeSubset):
				train_rows = datasetTrainRows
				datasetSize = train_rows
			else:
				train_rows =  int(datasetSize)
			eval_rows = int(0)
			train_stream = base_stream
			test_stream = None
			
		print(f"Train size: {train_rows:,}")
		print(f"Eval size: {eval_rows:,}")
		global datasetSizeRecord
		datasetSizeRecord = datasetSize

		dataset = DatasetDict({datasetSplitNameTrain: train_stream, datasetSplitNameTest: test_stream})

		return dataset

	def encode(batch):
		texts = batch["text"]
		if useNLPcharacterInput:
			out_ids = []
			for txt in texts:
				if useNLPcharacterInputBasic:
					txt = txt.lower()
				ids = []
				for ch in txt:
					if ch in _CHAR2ID:
						ids.append(_CHAR2ID[ch])
				out_ids.append(ids[:contextSizeMax])   # truncate, no pad yet
			return {"input_ids": out_ids}
		else:	
			global bert_tokenizer, bert_pad_id   # only used in BERT mode
			if bert_tokenizer is None:
				bert_tokenizer = AutoTokenizer.from_pretrained(bertModelName, use_fast=True)
				bert_pad_id	= bert_tokenizer.pad_token_id
				assert bert_pad_id == NLPcharacterInputPadTokenID
			enc = bert_tokenizer(texts, truncation=True, max_length=contextSizeMax, padding=False)
			if(debugOnlyPrintStreamedWikiArticleTitles):
				print(batch["title"])
			return enc   # already {"input_ids": [...], ...}
			
	def collate(batch):
		# batch_size==1 in your new set-up but keep it generic
		seqs   = [item for item in batch]
		pad_id = NLPcharacterInputPadTokenID
		max_L  = max(len(s) for s in seqs)
		padded = pt.full((len(seqs), max_L), pad_id, dtype=pt.long)
		for i, s in enumerate(seqs):
			padded[i, : len(s)] = s
		x = padded
		y = padded	 # no target yet (redundant)
		return x, y

	class RawSampleDataset(TorchIterableDataset):
		 """
		 Pass-through: yields one dict {'input_ids': Tensor[seq_len]} per article.
		 No left-shift / crop logic here any more.
		 """
		 def __init__(self, hf_iterable):
			 super().__init__()
			 self.hf_ds = hf_iterable

		 def __iter__(self):
			 for art in self.hf_ds:
				 ids = art["input_ids"]
				 if not isinstance(ids, pt.Tensor):
					 ids = pt.tensor(ids, dtype=pt.long)
				 yield ids
	
	def createDataLoaderNLP(dataset: "Dataset | HFDIterable"):
		"""Return DataLoader that yields (x, y) batches per the spec above."""
		
		ds_tok = dataset.map(encode, batched=True, remove_columns=dataset.column_names)
		ds_tok = ds_tok.with_format("torch")

		# If the result is map-style convert it, otherwise keep it as-is
		if isinstance(ds_tok, HFDIterable):
			ds_iter = ds_tok                      # already iterable -> nothing to do
		else:
			ds_iter = ds_tok.to_iterable_dataset()   # map-style -> convert

		ds = RawSampleDataset(ds_iter)
		
		loader = DataLoader(ds, batch_size=batchSize, collate_fn=collate, num_workers=numWorkers, pin_memory=pt.cuda.is_available())

		return loader

	if(useNLPcharacterInput):
	
		def ascii_printable_with_whitespace() -> list[str]:
			"""
			Return ASCII chars 0-127 with all control codes removed
			except the standard whitespace set:
				e.g. space (32), TAB (9), LF (10), [CR (13), VT (11), FF (12)]
			The order is stable: whitespace first, then 32-126 printable.
			"""
			# whitelist the whitespace control chars you want to keep
			whitespace_keep = {' ', '\t', '\n'}		#{' ', '\t', '\n', '\r', '\v', '\f'}

			chars = []
			# 0-127 inclusive
			for code in range(128):
				ch = chr(code)
				# keep if printable or whitelisted whitespace
				if ch.isprintable() or ch in whitespace_keep:
					chars.append(ch)
			return chars

		def _build_char_tables():
			if useNLPcharacterInputBasic:
				table = {c: i+1 for i, c in enumerate(NLPcharacterInputBasicSet)}  # 0 reserved for PAD (NLPcharacterInputPadTokenID)
				rev   = {i+1: c for i, c in enumerate(NLPcharacterInputBasicSet)}
			else:
				# Drop all control codes (0-31, 127) but keep whitespace
				allowed = ascii_printable_with_whitespace()
				assert len(allowed) == NLPcharacterInputSetLen-1	# -1 explanation; 0 reserved for PAD (NLPcharacterInputPadTokenID)
				table = {c: idx+1 for idx, c in enumerate(allowed)}	 #0 reserved for PAD (NLPcharacterInputPadTokenID)
				rev   = {idx+1: c for idx, c in enumerate(allowed)}
			return table, rev

		_CHAR2ID, _ID2CHAR = _build_char_tables()

"""EISANIpt_EISANImodelSequential.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model Sequential (sequentially activated neuronal input)

"""

import torch
from ANNpt_globalDefs import *
if(useDynamicGeneratedHiddenConnections):	#mandatory
	import EISANIpt_EISANImodelSequentialDynamic

def getWindowSize(self):
	windowSize = self.encodedFeatureSize//sequenceLength
	return windowSize
	
def calculateSegmentCompleteTokenWindowWidth(layerIdx):
	segmentCompleteTokenWindowWidth = layerIdx
	return segmentCompleteTokenWindowWidth

def calculateTime(batchIndex, slidingWindowIndex):
	currentTime = slidingWindowIndex
	#FUTURE: for contiguous datasets; ~ batchIndex*sequenceLength + slidingWindowIndex (see TSBNLPpt_dataLoaderOrdered.py for template)
	return currentTime
	
def sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, slidingWindowIndex, initActivation):

	device = initActivation.device
	currentActivationTime = calculateTime(batchIndex, slidingWindowIndex)

	#shift init layer activations (by number input bits) - always place new initActivation units at start of activation/time tensors;
	shiftUnits = getWindowSize(self)
	self.layerActivation[0] = torch.cat((torch.zeros_like(self.layerActivation[0][..., :shiftUnits]), self.layerActivation[0][..., :-shiftUnits]), dim=-1)
	self.layerActivationTime[0] = torch.cat((torch.zeros_like(self.layerActivationTime[0][..., :shiftUnits]), self.layerActivationTime[0][..., :-shiftUnits]), dim=-1)
	self.layerActivation[0][:, :shiftUnits] = initActivation
	self.layerActivationTime[0][:, :shiftUnits] = currentActivationTime
	#print("self.layerActivationTime[0][:, :shiftUnits*2] = ", self.layerActivationTime[0][:, :shiftUnits*2])
	
	for hiddenLayerIdx in range(self.config.numberOfHiddenLayers):
		if(debugSequentialSANIactivations):
			print("\nhiddenLayerIdx = ", hiddenLayerIdx)
		layerIdx = hiddenLayerIdx+1
		prevlayerIdx = layerIdx-1
		segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

		if(not dynamicGeneratedHiddenConnectionsAfterPropagating):
			dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx)

		#segment 1 activations;
		maxActivationTimeSegment1 = currentActivationTime
		layerSegment1Activation, layerSegment1Time, layerSegment1ActivationDistance, layerSegment1ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexProximal, device, maxActivationTimeSegment1)
		
		#segment 2 activations;
		if(useSequentialSANIactivationStrength and sequentialSANItimeInvariance):
			maxActivationTimeSegment2 = currentActivationTime - segmentCompleteTokenWindowWidth
			maxActivationRecallTime = calculateSegmentCompleteTokenWindowWidth(layerIdx)*sequentialSANItimeInvarianceFactor-1	#FUTURE: remove -1 condition (enable a small amount of timeInvariance for input layer)
			minActivationTimeSegment2 = max(0, maxActivationTimeSegment2-maxActivationRecallTime)
		else:
			maxActivationTimeSegment2 = currentActivationTime - segmentCompleteTokenWindowWidth
			minActivationTimeSegment2 = None
		layerSegment2Activation, layerSegment2Time, layerSegment2ActivationDistance, layerSegment2ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexDistal, device, maxActivationTimeSegment2, timeIndexMin=maxActivationTimeSegment2)
		layerActivation = torch.logical_and(layerSegment2Activation, layerSegment1Activation)

		if(useSequentialSANIactivationStrength):
			layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount = calculateActivationStrength(layerIdx, layerActivation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount)
		if(debugSequentialSANIactivations):
			print("\tdebugSequentialSANIactivations:")
			print("layerSegment1Activation = ", layerSegment1Activation)
			print("layerSegment2Activation = ", layerSegment2Activation)
			print("layerActivation = ", layerActivation)
			if(layerActivation.sum() > 0 and hiddenLayerIdx > 0): print("layerActivation.sum() > 0 and hiddenLayerIdx > 0 - successfully generating neurons across multiple layers")
			
		#update neuron activations;
		layerActivationNot = torch.logical_not(layerActivation)
		self.layerActivation[layerIdx] = torch.logical_or(self.layerActivation[layerIdx], layerActivation)
		self.layerActivationTime[layerIdx] = updateLayerData(self.layerActivationTime[layerIdx], layerActivation, layerActivationNot, currentActivationTime)	#reset the time values for current neuron activations
		if(useSequentialSANIactivationStrength):
			self.layerActivationDistance[layerIdx] = updateLayerData(self.layerActivationDistance[layerIdx], layerActivation, layerActivationNot, layerActivationDistance)
			self.layerActivationCount[layerIdx] = updateLayerData(self.layerActivationCount[layerIdx], layerActivation, layerActivationNot, layerActivationCount)
			#self.layerActivationStrength[layerIdx] = updateLayerData(self.layerActivationStrength[layerIdx], layerActivation, layerActivationNot, layerActivationStrength)
		
		if(dynamicGeneratedHiddenConnectionsAfterPropagating):
			dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx)

	layerActivationsList = self.layerActivation[1:]	#do not add input layer
	return layerActivationsList

def updateLayerData(lastActivationData, layerActivation, layerActivationNot, currentActivationData):
	#reset the data values for current neuron activations
	lastActivationData = lastActivationData * layerActivationNot.int()	
	lastActivationData = lastActivationData + (layerActivation.int()*currentActivationData)
	return lastActivationData

def maskLayerDataByTime(self, currentActivationTime: int, layerIdx: int, propData: str, timeIndexMax: int, timeIndexMin=None):
	
	activation = self.layerActivation[layerIdx]
	activationTime = self.layerActivationTime[layerIdx]
	if(layerIdx == 0):
		'''
		#select subset of layerActivations at timeIndex (of windowSize)
		timeOffsetMin = currentActivationTime-timeIndexMax
		if(timeIndexMin is None):
			timeIndexMin = timeIndexMax
		windowSize = getWindowSize(self)

		activation = self.layerActivation[layerIdx][:, timeOffsetMin:timeOffsetMin + (timeIndexMax-timeIndexMin+1)*windowSize]
		activationTime = self.layerActivationTime[layerIdx][:, timeOffsetMin:timeOffsetMin + (timeIndexMax-timeIndexMin+1)*windowSize]
		'''
		if(useSequentialSANIactivationStrength):
			activationDistance = torch.zeros_like(activation.int())	#initialise (internal) distance each input to zero
			activationCount = activation.int()		#treat each input count as 1
			#activationStrength = activation.float()	#not currently used (it is a derived parameter)
	else:
		if(useSequentialSANIactivationStrength):
			activationDistance = self.layerActivationDistance[layerIdx]
			activationCount = self.layerActivationCount[layerIdx]
			#activationStrength = self.layerActivationStrength[layerIdx]

	if(timeIndexMin is None):
		timeMask = (activationTime == timeIndexMax).float()
	else:
		timeMask = (torch.logical_and(activationTime <= timeIndexMax, activationTime >= timeIndexMin)).float()

	if(propData=="activation"):
		result = activation*timeMask	#filter activations for same time (at timeIndex)
	elif(propData=="activationTime"):
		result = activationTime*timeMask
	elif(propData=="activationDistance"):
		result = activationDistance*timeMask
	elif(propData=="activationCount"):
		result = activationCount*timeMask
	#elif(propData=="activationStrength"):	#not currently used (it is a derived parameter)
	#	result = activationStrength*timeMask

	return result

def compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime: int, hiddenLayerIdx: int, segmentIdx: int, device: torch.device, timeIndexMax = None, timeIndexMin = None) -> torch.Tensor:	
	if(debugSequentialSANIactivations):
		print("\tcompute_layer_sequentialSANI_allDataTypes = ")
		print("timeIndexMax = ", timeIndexMax)
		print("timeIndexMin = ", timeIndexMin)
	layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationStrength, layerSegmentXActivationCount = (None, None, None, None)
	layerSegmentXActivation = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activation", timeIndexMax, timeIndexMin)
	if(useSequentialSANIactivationStrength):
		layerSegmentXTime = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationTime", timeIndexMax, timeIndexMin)
		layerSegmentXActivationDistance = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationDistance", timeIndexMax, timeIndexMin)
		layerSegmentXActivationCount = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationCount", timeIndexMax, timeIndexMin)
	else:
		layerSegmentXTime = layerSegmentXActivationDistance = layerSegmentXActivationCount = None
	return layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationDistance, layerSegmentXActivationCount

def compute_layer_sequentialSANI(self, currentActivationTime: int, hiddenLayerIdx: int, segmentIdx: int, device: torch.device, propData: str, timeIndexMax = None, timeIndexMin = None) -> torch.Tensor:	
	layerIdx = hiddenLayerIdx+1
	prevlayerIdx = layerIdx-1
	prevActivation = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, propData, timeIndexMax, timeIndexMin)
	if(debugSequentialSANIactivations):
		print("propData = ", propData, ", prevActivation = ", prevActivation)
	
	if(useConnectionWeights):
		z_float = EISANIpt_EISANImodelSequentialDynamic.forwardProp(self, prevActivation, hiddenLayerIdx, segmentIdx, device)
	else:
		z_float = EISANIpt_EISANImodelSequentialDynamic.forwardProp(self, prevActivation, hiddenLayerIdx, segmentIdx)

	result = z_float
	#if(propData=="activation"): result = z_float >= segmentActivationThreshold	#not required as segmentActivationThreshold = 1

	return result

def calculateActivationStrength(layerIdx, layerActivation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount):
	segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

	if(debugSequentialSANIactivationsStrength):
		print("\tcalculateActivationStrength(): layerIdx = ", layerIdx)
		print("layerActivation = ", layerActivation)
		print("layerSegment1Time = ", layerSegment1Time)
		print("layerSegment2Time = ", layerSegment2Time)
		print("layerSegment1ActivationCount = ", layerSegment1ActivationCount)
		print("layerSegment2ActivationCount = ", layerSegment2ActivationCount)
		print("layerSegment1ActivationDistance = ", layerSegment1ActivationDistance)
		print("layerSegment2ActivationDistance = ", layerSegment2ActivationDistance)
		print("segmentCompleteTokenWindowWidth = ", segmentCompleteTokenWindowWidth)
		
	layerActivationCountNormalised = layerActivationProximityNormalised = layerActivationCount = layerActivationDistance = None

	layerActivationStrength = layerActivation.float()
	if(sequentialSANIsegmentsPartialActivation):
		layerActivationCount = layerSegment1ActivationCount + layerSegment2ActivationCount
		layerActivationCountNormalised = layerActivationCount.float() / count_predicted(layerIdx)
		layerActivationStrength = layerActivationStrength*layerActivationCountNormalised.float()
		if(debugSequentialSANIactivationsStrength):
			print("1 layerActivationStrength = ", layerActivationStrength)
	if(sequentialSANItimeInvariance):
		layerActivationDistance = layerSegment1ActivationDistance + layerSegment2ActivationDistance	#add distance of each segment
		layerActivationDistance = layerActivationDistance + (layerSegment1Time-layerSegment2Time)	#add distance between each segment
		layerActivationProximityNormalised = (1.0/layerActivationDistance.float()) * distance_predicted(layerIdx)
		layerActivationProximityNormalised[torch.isinf(layerActivationProximityNormalised)] = 0	#zero inf values
		layerActivationStrength = layerActivationStrength*layerActivationProximityNormalised
		if(debugSequentialSANIactivationsStrength):
			print("2 layerActivationStrength = ", layerActivationStrength)
	layerActivation = layerActivationStrength > segmentActivationFractionThreshold

	if(debugSequentialSANIactivationsStrength):
		print("layerActivationCount = ", layerActivationCount)
		print("layerActivationDistance = ", layerActivationDistance)
		print("layerActivationCountNormalised = ", layerActivationCountNormalised)
		print("layerActivationProximityNormalised = ", layerActivationProximityNormalised)
		print("layerActivation = ", layerActivation)
		print("layerActivationStrength = ", layerActivationStrength)

	return layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount

def count_predicted(layerIndex: int) -> int:
	'''
	layerIdx=0; 1
	layerIdx=1; 2
	layerIdx=2; 4
	layerIdx=3; 8
	layerIdx=4; 16
	'''	
	count = 2 ** layerIndex
	return count
	
def distance_predicted(layerIndex: int) -> int:	
	'''
	layerIdx=0; 0
	layerIdx=1; (0+0)+1
	layerIdx=2; (1+1)+2
	layerIdx=3; (2+2)+4
	layerIdx=4; (4+4)+8
	'''
	distance = 0 if layerIndex == 0 else 1 if layerIndex == 1 else 2 * (2 ** (layerIndex - 1))
	return distance

def dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx):
	layerIdx = hiddenLayerIdx+1
	if(trainOrTest and useDynamicGeneratedHiddenConnections):
		timeSeg0 = currentActivationTime
		timeSeg1 = currentActivationTime-layerIdx	#assume network diverges by 1 unit every layer
		prevlayerIdx = layerIdx-1
		prevActivationSeg0 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg0)
		prevActivationSeg1 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg1)
		#print("prevlayerIdx = ", prevlayerIdx)
		#print("timeSeg0 = ", timeSeg0)
		#print("timeSeg1 = ", timeSeg1)
		#print("prevActivationSeg0 = ", prevActivationSeg0)
		#print("prevActivationSeg1 = ", prevActivationSeg1)
		EISANIpt_EISANImodelSequentialDynamic.sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx, prevActivationSeg0, prevActivationSeg1)

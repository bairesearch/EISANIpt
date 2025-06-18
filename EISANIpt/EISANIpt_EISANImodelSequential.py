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

def calculateActivationStrength(layerIdx, layerActivation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount):
	segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

	layerActivationCount = layerActivationDistance = None
	layerActivationCountNormalised = layerActivationProximityNormalised = None

	layerActivationStrength = layerActivation.float()
	if(sequentialSANIsegmentsPartialActivation):
		layerActivationCount = layerSegment1ActivationCount + layerSegment2ActivationCount
		layerActivationCountNormalised = layerActivationCount.float() / segmentCompleteTokenWindowWidth	#CHECKTHIS noramlisation method	#note the activation count is normalised by layerIdx (segmentCompleteTokenWindowWidth)
		#layerActivationCountNormalised = layerActivationCountNormalised * layerActivation.float()	#zero bad values
		layerActivationStrength = layerActivationStrength*layerActivationCount.float()	#note the activation count is not currently normalised by layerIdx - activation strength used for output predictions will be biased by layer index; CONSIDER: /layerIdx
	if(sequentialSANItimeInvariance):
		layerActivationDistance = layerSegment1ActivationDistance + layerSegment2ActivationDistance	#add distance of each segment
		layerActivationDistance = layerActivationDistance + (layerSegment2Time-layerSegment1Time)	#add distance between each segment
		layerActivationProximityNormalised = 1.0/layerActivationDistance.float() / segmentCompleteTokenWindowWidth 	#CHECKTHIS noramlisation method	#note the activation proximity is normalised by layerIdx (segmentCompleteTokenWindowWidth)
		layerActivationProximityNormalised[torch.isinf(layerActivationProximityNormalised)] = 0	#zero inf values
		layerActivationStrength = layerActivationStrength*layerActivationProximityNormalised
	layerActivation = layerActivationStrength/segmentCompleteTokenWindowWidth > segmentActivationFractionThreshold	#CHECKTHIS threshold

	if(debugSequentialSANIweightedActivations):
		print("\ncalculateActivationStrength(): layerIdx = ", layerIdx)
		print("layerActivationCount = ", layerActivationCount)
		print("layerActivationDistance = ", layerActivationDistance)
		print("layerActivationCountNormalised = ", layerActivationCountNormalised)
		print("layerActivationProximityNormalised = ", layerActivationProximityNormalised)
		print("layerActivation = ", layerActivation)
		print("layerActivationStrength = ", layerActivationStrength)

	return layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount

def updateLayerData(lastActivationData, layerActivation, layerActivationNot, currentActivationData):
	#reset the data values for current neuron activations
	lastActivationData = lastActivationData * layerActivationNot.int()	
	lastActivationData = lastActivationData + (layerActivation.int()*currentActivationData)
	return lastActivationData

def sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, slidingWindowIndex, initActivation):

	device = initActivation.device
	currentActivationTime = calculateTime(batchIndex, slidingWindowIndex)

	#shift init layer activations (by number input bits) - always place new initActivation units at start of activation/time tensors;
	shiftUnits = getWindowSize(self)
	self.layerActivation[0] = torch.cat((torch.zeros_like(self.layerActivation[0][..., :shiftUnits]), initActivation), dim=-1)
	self.layerActivation[0][:, :shiftUnits] = initActivation
	self.layerActivationTime[0][:, :shiftUnits] = currentActivationTime

	for hiddenLayerIdx in range(self.config.numberOfHiddenLayers):
		layerIdx = hiddenLayerIdx+1
		prevlayerIdx = layerIdx-1
		segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

		#segment 1 activations;
		layerSegment1Activation, layerSegment1Time, layerSegment1ActivationDistance, layerSegment1ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexProximal, device, currentActivationTime)
		
		#segment 2 activations;
		if(sequentialSANItimeInvariance):
			maxActivationTimeSegment2 = currentActivationTime
			maxActivationRecallTime = calculateSegmentCompleteTokenWindowWidth(layerIdx)*sequentialSANItimeInvarianceFactor
			minActivationTimeSegment2 = max(0, currentActivationTime-maxActivationRecallTime)
		else:
			maxActivationTimeSegment2 = currentActivationTime - segmentCompleteTokenWindowWidth
			minActivationTimeSegment2 = None
		layerSegment2Activation, layerSegment2Time, layerSegment2ActivationDistance, layerSegment2ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexDistal, device, maxActivationTimeSegment2, timeIndexMin=maxActivationTimeSegment2)
		layerActivation = torch.logical_and(layerSegment2Activation, layerSegment1Activation)
		if(sequentialSANIweightedActivations):
			layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount = calculateActivationStrength(layerIdx, layerActivation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount)

		#update neuron activations;
		layerActivationNot = torch.logical_not(layerActivation)
		self.layerActivation[layerIdx] = torch.logical_or(self.layerActivation[layerIdx], layerActivation)
		self.layerActivationTime[layerIdx] = updateLayerData(self.layerActivationTime[layerIdx], layerActivation, layerActivationNot, currentActivationTime)	#reset the time values for current neuron activations
		if(sequentialSANIweightedActivations):
			self.layerActivationDistance[layerIdx] = updateLayerData(self.layerActivationDistance[layerIdx], layerActivation, layerActivationNot, layerActivationDistance)
			self.layerActivationCount[layerIdx] = updateLayerData(self.layerActivationCount[layerIdx], layerActivation, layerActivationNot, layerActivationCount)
			#self.layerActivationStrength[layerIdx] = updateLayerData(self.layerActivationStrength[layerIdx], layerActivation, layerActivationNot, layerActivationStrength)

		if(trainOrTest and useDynamicGeneratedHiddenConnections):
			timeSeg0 = currentActivationTime
			timeSeg1 = currentActivationTime-layerIdx	#assume network diverges by 1 unit every layer
			prevlayerIdx = layerIdx-1
			prevActivationSeg0 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg0)
			prevActivationSeg1 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg1)
			EISANIpt_EISANImodelSequentialDynamic.sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx, prevActivationSeg0, prevActivationSeg1)

	layerActivationsList = self.layerActivation[1:]	#do not add input layer
	return layerActivationsList

def calculateTime(batchIndex, slidingWindowIndex):
	currentTime = slidingWindowIndex
	#FUTURE: for contiguous datasets; ~ batchIndex*sequenceLength + slidingWindowIndex (see TSBNLPpt_dataLoaderOrdered.py for template)
	return currentTime

def maskLayerDataByTime(self, currentActivationTime: int, layerIdx: int, propData: str, timeIndexMax: int, timeIndexMin=None):
	if(layerIdx == 0):
		#select subset of layerActivations at timeIndex (of windowSize)
		timeOffsetMin = currentActivationTime-timeIndexMax
		if(timeIndexMin is None):
			timeIndexMin = timeIndexMax
		windowSize = getWindowSize(self)

		activation = self.layerActivation[layerIdx][:, timeOffsetMin:timeOffsetMin + (timeIndexMax-timeIndexMin+1)*windowSize]
		activationTime = self.layerActivationTime[layerIdx][:, timeOffsetMin:timeOffsetMin + (timeIndexMax-timeIndexMin+1)*windowSize]
		if(sequentialSANIweightedActivations):
			activationDistance = torch.zeros_like(activation.int())	#initialise (internal) distance each input to zero
			activationCount = activation.int()		#treat each input count as 1
			#activationStrength = activation.float()	#not currently used (it is a derived parameter)
	else:
		activation = self.layerActivation[layerIdx]
		activationTime = self.layerActivationTime[layerIdx]
		if(sequentialSANIweightedActivations):
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
	layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationStrength, layerSegmentXActivationCount = (None, None, None, None)
	layerSegmentXActivation = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activation", timeIndexMax, timeIndexMin)
	if(sequentialSANIweightedActivations):
		layerSegmentXTime = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationTime", timeIndexMax, timeIndexMin)
		layerSegmentXActivationDistance = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationDistance", timeIndexMax, timeIndexMin)
		layerSegmentXActivationCount = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationCount", timeIndexMax, timeIndexMin)
	return layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationDistance, layerSegmentXActivationCount

def compute_layer_sequentialSANI(self, currentActivationTime: int, hiddenLayerIdx: int, segmentIdx: int, device: torch.device, propData: str, timeIndexMax = None, timeIndexMin = None) -> torch.Tensor:	
	layerIdx = hiddenLayerIdx+1
	prevlayerIdx = layerIdx-1
	prevActivation = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, propData, timeIndexMax, timeIndexMin)

	if(useConnectionWeights):
		z_float = EISANIpt_EISANImodelSequentialDynamic.forwardProp(self, prevActivation, hiddenLayerIdx, segmentIdx, device)
	else:
		z_float = EISANIpt_EISANImodelSequentialDynamic.forwardProp(self, prevActivation, hiddenLayerIdx, segmentIdx)

	result = z_float
	#if(propData=="activation"): result = z_float >= segmentActivationThreshold	#not required as segmentActivationThreshold = 1

	return result

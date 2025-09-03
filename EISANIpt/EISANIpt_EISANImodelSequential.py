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

	layerActivationsList = []
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
		if(debugSequentialSANIactivationsLoops):
			print("\nhiddenLayerIdx = ", hiddenLayerIdx)
		layerIdx = hiddenLayerIdx+1
		prevlayerIdx = layerIdx-1
		#segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

		if(not generateConnectionsAfterPropagating):
			dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx)

		#segment 1 activations;
		maxActivationTimeSegment1, maxActivationTimeSegment2 = calculateSegmentTimes(currentActivationTime, layerIdx)
		layerSegment1Activation, layerSegment1Time, layerSegment1ActivationDistance, layerSegment1ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexProximal, device, maxActivationTimeSegment1)
		
		#segment 2 activations;
		minActivationTimeSegment2 = None
		if(self.sequentialSANItimeInvariance):
			maxActivationRecallTimeInvariance = calculateMaxActivationRecallTimeInvariance(layerIdx)
			if(not inputLayerTimeInvariance and hiddenLayerIdx==0):
				maxActivationRecallTimeInvariance = 0	#disable timeInvariance for input layer
			if(debugSequentialSANItimeInvarianceDisable):
				maxActivationRecallTimeInvariance = 0	#disable time invariance for temp debug, but still print all time invariance (distance/proximity) calculations!
			minActivationTimeSegment2 = max(0, maxActivationTimeSegment2-maxActivationRecallTimeInvariance)
			
		layerSegment2Activation, layerSegment2Time, layerSegment2ActivationDistance, layerSegment2ActivationCount = compute_layer_sequentialSANI_allDataTypes(self, currentActivationTime, hiddenLayerIdx, sequentialSANIsegmentIndexDistal, device, maxActivationTimeSegment2, timeIndexMin=minActivationTimeSegment2)

		if(self.useSequentialSANIactivationStrength):
			layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount = calculateActivationStrength(layerIdx, layerSegment1Activation, layerSegment2Activation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount)
		else:
			layerActivation = torch.logical_and(layerSegment1Activation, layerSegment2Activation)
			layerActivationStrength = layerActivation
			
		if(debugSequentialSANIactivations):
			print("\tdebugSequentialSANIactivations:")
			print("layerSegment1Activation = ", layerSegment1Activation)
			print("layerSegment2Activation = ", layerSegment2Activation)
			print("layerActivation = ", layerActivation)
			print("layerActivation.shape = ", layerActivation.shape)
			print("layerActivation.count_nonzero = ", layerActivation.count_nonzero(dim=1)) 
		if(debugSequentialSANIpropagationVerify):
			if(layerActivation.sum() > 0 and hiddenLayerIdx > 0): 
				printe("layerActivation.sum() > 0 and hiddenLayerIdx > 0 - successfully generating neurons across multiple layers")
					
		if(limitConnections and limitHiddenConnections):
			self.hiddenNeuronUsage[hiddenLayerIdx] = self.hiddenNeuronUsage[hiddenLayerIdx] + layerActivationStrength.sum(dim=0)	#sum across batch dim	#or layerActivation.int().sum(dim=0)?
				
		#update neuron activations;
		layerActivationNot = torch.logical_not(layerActivation)
		self.layerActivation[layerIdx] = torch.logical_or(self.layerActivation[layerIdx], layerActivation)
		self.layerActivationTime[layerIdx] = updateLayerData(self.layerActivationTime[layerIdx], layerActivation, layerActivationNot, currentActivationTime)	#reset the time values for current neuron activations
		if(self.useSequentialSANIactivationStrength):
			self.layerActivationDistance[layerIdx] = updateLayerData(self.layerActivationDistance[layerIdx], layerActivation, layerActivationNot, layerActivationDistance)
			self.layerActivationCount[layerIdx] = updateLayerData(self.layerActivationCount[layerIdx], layerActivation, layerActivationNot, layerActivationCount)
			#self.layerActivationStrength[layerIdx] = updateLayerData(self.layerActivationStrength[layerIdx], layerActivation, layerActivationNot, layerActivationStrength)
		self.layerActivationStrength = layerActivationStrength	#required for dynamicallyGenerateLayerNeurons	#temporary var
		
		if(generateConnectionsAfterPropagating):
			dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx)
		
		if(evalOnlyUsingTimeInvariance and not trainOrTest):
			if(evalStillTrainOutputConnections):
				layerActivationsList.append(self.layerActivationStrength)
			else:
				layerActivationsList.append(layerActivation)	#always pass boolean hidden activations to output prediction layer (even when self.sequentialSANItimeInvariance==True during eval), as model was trained expecting boolean inputs to output layer
		else:
			layerActivationsList.append(self.layerActivationStrength)
	#orig; layerActivationsList = self.layerActivation[1:]	#do not add input layer

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
		if(self.useSequentialSANIactivationStrength):
			activationDistance = torch.zeros_like(activation.int())	#initialise (internal) distance each input to zero
			activationCount = activation.int()		#treat each input count as 1
			#activationStrength = activation.float()	#not currently used (it is a derived parameter)
	else:
		if(self.useSequentialSANIactivationStrength):
			activationDistance = self.layerActivationDistance[layerIdx]
			activationCount = self.layerActivationCount[layerIdx]
			#activationStrength = self.layerActivationStrength[layerIdx]

	if(timeIndexMin is None):
		timeMask = (activationTime == timeIndexMax).float()
	else:
		timeMask = (torch.logical_and(activationTime <= timeIndexMax, activationTime >= timeIndexMin)).float()
		if(debugSequentialSANItimeInvarianceVerify):
			print("timeMask.count_nonzero = ", timeMask.count_nonzero(dim=1))
			maskedTimes = activationTime*timeMask
			print("maskedTimes[maskedTimes != 0].tolist() = ", maskedTimes[maskedTimes != 0].tolist())

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
	if(self.useSequentialSANIactivationStrength):
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
		print("prevActivation.count_nonzero = ", prevActivation.count_nonzero(dim=1))
	
	z_float = EISANIpt_EISANImodelSequentialDynamic.forwardProp(self, prevActivation, hiddenLayerIdx, segmentIdx)

	result = z_float
	#if(propData=="activation"): result = z_float >= segmentActivationThreshold	#not required as segmentActivationThreshold = 1

	return result

def calculateActivationStrength(layerIdx, layerSegment1Activation, layerSegment2Activation, layerSegment1Time, layerSegment2Time, layerSegment1ActivationDistance, layerSegment2ActivationDistance, layerSegment1ActivationCount, layerSegment2ActivationCount):
	#segmentCompleteTokenWindowWidth = calculateSegmentCompleteTokenWindowWidth(layerIdx)

	if(debugSequentialSANIactivationsStrength):
		print("\tcalculateActivationStrength(): layerIdx = ", layerIdx)
		print("layerActivation = ", layerActivation)
		print("layerSegment1Time = ", layerSegment1Time)
		print("layerSegment2Time = ", layerSegment2Time)
		print("layerSegment1ActivationCount = ", layerSegment1ActivationCount)
		print("layerSegment2ActivationCount = ", layerSegment2ActivationCount)
		print("layerSegment1ActivationDistance = ", layerSegment1ActivationDistance)
		print("layerSegment2ActivationDistance = ", layerSegment2ActivationDistance)
		#print("segmentCompleteTokenWindowWidth = ", segmentCompleteTokenWindowWidth)
		
	layerActivationCountNormalised = layerActivationProximityNormalised = layerActivationCount = layerActivationDistance = None

	if(sequentialSANIsegmentRequirement == "both"):
		layerActivationStrength = torch.logical_and(layerSegment1Activation, layerSegment2Activation)
	elif(sequentialSANIsegmentRequirement == "any"):
		layerActivationStrength = torch.logical_or(layerSegment1Activation, layerSegment2Activation)
		#filter seg1/seg2 activation properties by layerSegment1Activation/layerSegment2Activation;
		layerSegment1Time = layerSegment1Time*layerSegment1Activation
		layerSegment1ActivationDistance = layerSegment1ActivationDistance*layerSegment1Activation
		layerSegment1ActivationCount = layerSegment1ActivationCount*layerSegment1Activation
		layerSegment2Time = layerSegment2Time*layerSegment2Activation
		layerSegment2ActivationDistance = layerSegment2ActivationDistance*layerSegment2Activation
		layerSegment2ActivationCount = layerSegment2ActivationCount*layerSegment2Activation
	elif(sequentialSANIsegmentRequirement == "first"):
		layerActivationStrength = layerSegment1Activation
		#filter seg2 activation properties by layerSegment2Activation;
		layerSegment2Time = layerSegment2Time*layerSegment2Activation
		layerSegment2ActivationDistance = layerSegment2ActivationDistance*layerSegment2Activation
		layerSegment2ActivationCount = layerSegment2ActivationCount*layerSegment2Activation
	elif(sequentialSANIsegmentRequirement == "none"):	
		printe("sequentialSANIsegmentRequirement == none does not currently work as seg1/seg2 activation properties (time/distance/count) are only currently valid when layerActivation values are true")
		layerActivationStrength = torch.ones_like(layerSegment1Activation).float()
		#no prior dependency layerActivation dependency (no non-linear activation functions)

	layerActivationCount = layerSegment1ActivationCount + layerSegment2ActivationCount + 1
	#layerActivationCount = layerActivationCount - overlappingCountBetweenSegments
	layerActivationCountNormalised = layerActivationCount.float() / count_predicted(layerIdx)
	if(debugSequentialSANIactivationsStrength):
		print("\tsequentialSANIsegmentsPartialActivationCount:")
		print("layerActivationCount = ",layerActivationCount)
		print("count_predicted(layerIdx) = ", count_predicted(layerIdx))
		print("layerActivationCountNormalised = ", layerActivationCountNormalised)
	if(sequentialSANIsegmentsPartialActivationCount):
		layerActivationStrength = layerActivationStrength*layerActivationCountNormalised.float()

	layerActivationDistance = layerSegment1ActivationDistance + layerSegment2ActivationDistance	#add distance of each segment
	layerActivationDistance = layerActivationDistance + (layerSegment1Time-layerSegment2Time)	#add distance between each segment
	layerActivationProximityNormalised = (1.0/layerActivationDistance.float()) * distance_predicted(layerIdx)
	layerActivationProximityNormalised[torch.isinf(layerActivationProximityNormalised)] = 0	#zero inf values
	if(debugSequentialSANIactivationsStrength):
		print("\tsequentialSANIsegmentsPartialActivationDistance:")
		print("layerActivationDistance = ", layerActivationDistance)
		print("layerSegment1ActivationDistance + layerSegment2ActivationDistance = ", layerSegment1ActivationDistance + layerSegment2ActivationDistance)
		print("(layerSegment1Time-layerSegment2Time) = ", (layerSegment1Time-layerSegment2Time))
		print("distance_predicted(layerIdx) = ", distance_predicted(layerIdx))
		print("layerActivationProximityNormalised = ", layerActivationProximityNormalised)
	if(sequentialSANIsegmentsPartialActivationDistance):
		layerActivationStrength = layerActivationStrength*layerActivationProximityNormalised

	layerActivation = layerActivationStrength >= segmentActivationFractionThreshold
	layerActivationStrength = layerActivationStrength*layerActivation.float()
	
	if(sequentialSANIinhibitoryTopkSelection):
		if(debugSequentialSANIinhibitoryTopkSelection):
			print("\tsequentialSANIinhibitoryTopkSelection:")
			print("layerActivationStrength.shape = ", layerActivationStrength.shape)
			print("pre active neurons = ", layerActivationStrength.count_nonzero(dim=1)) 
			#print("pre active neuron strengths = ", layerActivationStrength.count_nonzero(dim=1)) 
		k = int(layerActivationStrength.shape[1]*sequentialSANIinhibitoryTopkSelectionKfraction)
		layerActivationTopKmask = generateTopkMask(layerActivationStrength, k)
		layerActivation = layerActivation*mask	#zero out everything except the chosen activations
		layerActivationStrength = layerActivationStrength*mask	#zero out everything except the chosen activations
		#print("post active neuron strengths = ", layerActivationStrength.count_nonzero(dim=1)) 

	if(debugSequentialSANIactivationsStrength):
		print("\tlayerActivation = ", layerActivation)
		print("layerActivationStrength = ", layerActivationStrength)

	return layerActivation, layerActivationStrength, layerActivationDistance, layerActivationCount

def generateTopkMask(x: torch.Tensor, k: int) -> torch.Tensor:
	"""
	Keep only the top-k activations in each row of a (B, N) tensor.

	Args:
		x (torch.Tensor): Input tensor of shape (B, N).
		k (int)          : Number of top elements to keep per row.

	Returns:
		torch.Tensor: Tensor of the same shape as `x` where the
		              largest `k` values in every row are preserved
		              and all other entries are set to 0.
	"""
	if k <= 0:
		return torch.zeros_like(x)

	# 1. Find indices of the top-k values along dim 1 (per batch item)
	_, idx = torch.topk(x, k=min(k, x.size(1)), dim=1)

	# 2. Build a boolean mask with True at those indices
	mask = torch.zeros_like(x, dtype=torch.bool)
	mask.scatter_(1, idx, True)

def calculate_segment_times_diff(layerIndex: int) -> int:
	if(sequentialSANIoverlappingSegments):
		r'''
		\/\/\/
		 \/\/
		  \/
		layerIdx=0; = [N/A]
		layerIdx=1; = 1
		layerIdx=2; = 1
		layerIdx=3; = 1
		'''	
		diff = 1
	else:
		r'''
		\/  \/  \/  \/
		 \  /    \  /
		   \      /
		layerIdx=0; [N/A]
		layerIdx=1; 1
		layerIdx=2; 2
		layerIdx=3; 4
		layerIdx=4; 8
		'''
		diff = 2 ** (layerIndex - 1)
	return diff
	
def count_predicted(layerIndex: int) -> int:
	r'''
	if(sequentialSANIoverlappingSegments):
	#counts overlapping nodes as separate nodes
	
	\/\/\/
	 \/\/
	  \/
	
	else:
	
	\/  \/  \/  \/
	 \  /    \  /
	   \      /
	   
	layerIdx=0; 1
	layerIdx=1; 3
	layerIdx=2; 7
	layerIdx=3; 15
	layerIdx=4; 31
	'''
	count =  (2 ** (layerIndex+1) - 1)
	return count
	
def distance_predicted(layerIndex: int) -> int:	
	if(sequentialSANIoverlappingSegments):
		r'''
		\/\/\/
		 \/\/
		  \/
		layerIdx=0; 0
		layerIdx=1; (0+0)+1 = 1		#layerSegment1or2ActivationDistance=0, (layerSegment1Time-layerSegment2Time)=1
		layerIdx=2; (1+1)+1 = 3		#layerSegment1or2ActivationDistance=1, (layerSegment1Time-layerSegment2Time)=1
		layerIdx=3; (3+3)+1 = 7		#layerSegment1or2ActivationDistance=3, (layerSegment1Time-layerSegment2Time)=1
		layerIdx=4; (7+7)+1 = 15
		layerIdx=5; (15+15)+1 = 31
		'''
		distance = (2 ** layerIndex - 1)
	else:
		r'''
		\/  \/  \/  \/
		 \  /    \  /
		   \      /
		layerIdx=0; 0
		layerIdx=1; (0+0)+1 = 1		#layerSegment1or2ActivationDistance=0, (layerSegment1Time-layerSegment2Time)=1
		layerIdx=2; (1+1)+2 = 4		#layerSegment1or2ActivationDistance=1, (layerSegment1Time-layerSegment2Time)=2
		layerIdx=3; (4+4)+4 = 12	#layerSegment1or2ActivationDistance=4, (layerSegment1Time-layerSegment2Time)=4
		layerIdx=4; (12+12)+8 = 32
		'''
		distance = 0 if layerIndex == 0 else layerIndex * (2 ** (layerIndex - 1))
	return distance

def calculateMaxActivationRecallTimeInvariance(layerIdx):
	if(sequentialSANIoverlappingSegments):
		maxActivationRecallTimeInvariance = int(layerIdx*sequentialSANItimeInvarianceFactor)	#heuristic
	else:
		maxActivationRecallTimeInvariance = int(calculate_segment_times_diff(layerIdx)*sequentialSANItimeInvarianceFactor)
	return maxActivationRecallTimeInvariance

def calculateSegmentTimes(currentActivationTime, layerIdx):
	timeSeg0 = currentActivationTime
	timeSeg1 = currentActivationTime - calculate_segment_times_diff(layerIdx)	#assume network diverges by 1 unit every layer
	return timeSeg0, timeSeg1
			
def dynamicallyGenerateLayerNeurons(self, trainOrTest, currentActivationTime, hiddenLayerIdx):
	layerIdx = hiddenLayerIdx+1
	if(trainOrTest and useDynamicGeneratedHiddenConnections):
		timeSeg0, timeSeg1 = calculateSegmentTimes(currentActivationTime, layerIdx)
		prevlayerIdx = layerIdx-1
		prevActivationSeg0 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg0)
		prevActivationSeg1 = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, "activation", timeSeg1)
		if(debugDynamicallyGenerateLayerNeurons):
			print("\tdebugDynamicallyGenerateLayerNeurons:")
			print("prevlayerIdx = ", prevlayerIdx)
			print("timeSeg0 = ", timeSeg0)
			print("timeSeg1 = ", timeSeg1)
			print("prevActivationSeg0 = ", prevActivationSeg0)
			print("prevActivationSeg1 = ", prevActivationSeg1)
		if(debugSequentialSANIactivationsMemory):
			print("\tdebugDynamicallyGenerateLayerNeurons:")
			print("self.layerActivation[layerIdx].shape = ", self.layerActivation[layerIdx].shape)
			#print("dynamicallyGenerateLayerNeurons: self.layerActivation[layerIdx].sum() = ", self.layerActivation[layerIdx].sum())
			print("prevActivationSeg0.sum() = ",prevActivationSeg0.sum())
			print("prevActivationSeg1.sum() = ",prevActivationSeg1.sum())

		EISANIpt_EISANImodelSequentialDynamic.sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx, prevActivationSeg0, prevActivationSeg1)
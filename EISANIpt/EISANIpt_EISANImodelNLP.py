"""EISANIpt_EISANImodelNLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model NLP

"""

import torch
from ANNpt_globalDefs import *
from transformers import AutoModel
import EISANIpt_EISANImodelDynamic

if(useTokenEmbedding):
	model = AutoModel.from_pretrained(bertModelName).eval().to(device)   # embeddingSize-dim vectors

if(useSequentialSANI):

	def sequentialSANI_dynamic_hidden_growth_pairwise(										# pylint: disable=too-many-arguments
			self,
			hiddenLayerIdx: int,
			prevActivationSeg0: torch.Tensor,		# (B, prevSize) activations feeding segment segIndex0
			prevActivationSeg1: torch.Tensor,		# (B, prevSize) activations feeding segment segIndex1
			device: torch.device,
			segIndex0: int = 0,
			segIndex1: int = 1,
	) -> None:
		"""
		Simplified growth rule: create a new hidden neuron for every combination
		of active presynaptic neurons (idx0 \u2208 prevActivationSeg0, idx1 \u2208 prevActivationSeg1).

		Each new neuron receives exactly one synapse on each of the two given
		segments:
			\u2022 segment segIndex0 \u2190 idx0 (weight = +1)
			\u2022 segment segIndex1 \u2190 idx1 (weight = +1)

		Growth is skipped if the (idx0, idx1) pair has been used before or if no
		free neurons remain that are un-assigned on *both* segments.
		"""

		# ------------------------------------------------------------------
		# 1. Identify currently active presynaptic indices for each segment
		# ------------------------------------------------------------------
		active0 = (prevActivationSeg0 > 0).any(dim=0).nonzero(as_tuple=True)[0]	# 1-D LongTensor
		active1 = (prevActivationSeg1 > 0).any(dim=0).nonzero(as_tuple=True)[0]

		if active0.numel() == 0 or active1.numel() == 0:
			return		# nothing to pair

		# ------------------------------------------------------------------
		# 2. Prepare bookkeeping structures (create once on first call)
		# ------------------------------------------------------------------
		pairSigDict = self.hiddenNeuronPairSignatures[hiddenLayerIdx]

		# ------------------------------------------------------------------
		# 3. List neurons still free on *both* segments
		# ------------------------------------------------------------------
		freeMask = (~self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex0] &
					~self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex1])
		freeList = freeMask.nonzero(as_tuple=True)[0]				# 1-D LongTensor

		if freeList.numel() == 0:
			if self.training:
				print(f"Warning: no free neurons left for pairwise growth in layer {hiddenLayerIdx}")
			return

		newIdxCursor = 0		# walk through freeList as we allocate

		# ------------------------------------------------------------------
		# 4. Generate a hidden neuron for every (idx0, idx1) pair
		# ------------------------------------------------------------------
		for idx0 in active0.tolist():
			for idx1 in active1.tolist():

				# --- 4a: uniqueness check -------------------------------------------------
				pairSig = f"{idx0},{idx1}"
				if pairSig in pairSigDict:
					continue		# pair already realised

				# --- 4b: ensure we still have capacity -----------------------------------
				if newIdxCursor >= freeList.numel():
					if self.training:
						print(f"Warning: ran out of neurons in layer {hiddenLayerIdx} while forming pairs")
					return

				newNeuronIdx = freeList[newIdxCursor].item()
				newIdxCursor += 1

				# --- 4c: mark the neuron as assigned on both segments --------------------
				self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex0, newNeuronIdx] = True
				self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex1, newNeuronIdx] = True

				# --- 4d: connect idx0 on segIndex0, idx1 on segIndex1 --------------------
				self._assign_single_connection(hiddenLayerIdx, segIndex0, newNeuronIdx, idx0, device)
				self._assign_single_connection(hiddenLayerIdx, segIndex1, newNeuronIdx, idx1, device)

				# --- 4e: remember the signature so we never duplicate this pair ----------
				pairSigDict[pairSig] = True


	# --------------------------------------------------------------------------
	# Helper: add exactly one synapse (weight = +1 | True) to a sparse or dense
	#		 connection matrix in-place, then write the updated matrix back.
	# --------------------------------------------------------------------------
	def _assign_single_connection(
			self,
			hiddenLayerIdx: int,
			segIdx: int,
			neuronIdx: int,
			prevIdx: int,
			device: torch.device,
	) -> None:

		mat = self.hiddenConnectionMatrix[hiddenLayerIdx][segIdx]

		if self.config.useSparseMatrix:
			existing_indices = mat._indices()
			existing_values  = mat._values()

			new_indices = torch.tensor([[neuronIdx], [prevIdx]], dtype=torch.long, device=device)
			new_values  = torch.ones(1, dtype=existing_values.dtype, device=device)

			mat = torch.sparse_coo_tensor(
				torch.cat([existing_indices, new_indices], dim=1),
				torch.cat([existing_values,  new_values ], dim=0),
				mat.size(),
				device=device,
			).coalesce()
		else:
			mat[neuronIdx, prevIdx] = 1.0

		self.hiddenConnectionMatrix[hiddenLayerIdx][segIdx] = mat.to(device)


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
		print("currentActivationTime = ", currentActivationTime)
		
		#shift init layer activations (by number input bits) - always place new initActivation units at start of activation/time tensors;
		shiftUnits = windowSize = self.encodedFeatureSize
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
				sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx, prevActivationSeg0, prevActivationSeg1, device)

		layerActivationsList = list(self.layerActivation)
		return layerActivationsList

	def calculateTime(batchIndex, slidingWindowIndex):
		currentTime = slidingWindowIndex
		#FUTURE: for contiguous datasets; ~ batchIndex*sequenceLength + slidingWindowIndex (see TSBNLPpt_dataLoaderOrdered.py for template)
		return currentTime

	def maskLayerDataByTime(self, currentActivationTime: int, layerIdx: int, propData: str, timeIndexMax: int, timeIndexMin=None):
		if(layerIdx == 0):
			#select subset of layerActivations at timeIndex (of windowSize)
			print("timeIndexMax = ", timeIndexMax)
			timeOffsetMin = currentActivationTime-timeIndexMax
			if(timeIndexMin is None):
				timeIndexMin = timeIndexMax
			windowSize = self.encodedFeatureSize
			
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
		print("1 timeIndexMax = ", timeIndexMax)
		layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationStrength, layerSegmentXActivationCount = (None, None, None, None)
		layerSegmentXActivation = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activation", timeIndexMax, timeIndexMin)
		if(sequentialSANIweightedActivations):
			layerSegmentXTime = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationTime", timeIndexMax, timeIndexMin)
			layerSegmentXActivationDistance = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationDistance", timeIndexMax, timeIndexMin)
			layerSegmentXActivationCount = compute_layer_sequentialSANI(self, currentActivationTime, hiddenLayerIdx, segmentIdx, device, "activationCount", timeIndexMax, timeIndexMin)
		return layerSegmentXActivation, layerSegmentXTime, layerSegmentXActivationDistance, layerSegmentXActivationCount
				
	def compute_layer_sequentialSANI(self, currentActivationTime: int, hiddenLayerIdx: int, segmentIdx: int, device: torch.device, propData: str, timeIndexMax = None, timeIndexMin = None) -> torch.Tensor:	
		print("2 timeIndexMax = ", timeIndexMax)
		layerIdx = hiddenLayerIdx+1
		prevlayerIdx = layerIdx-1
		prevActivation = maskLayerDataByTime(self, currentActivationTime, prevlayerIdx, propData, timeIndexMax, timeIndexMin)

		# prevActivation is torch.int8 (0 or 1)
		weight = self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIdx].to(device)

		dev	= prevActivation.device
		# weight = self.hiddenConnectionMatrix[hiddenLayerIdx].to(dev) # Already done above

		if useSparseMatrix:
			# Called only when self.useEIneurons is False.
			# Sparse bool weights: True is +1, False is -1.
			weight = weight.coalesce()
			indices = weight.indices()
			values = weight.values() # bool
			numeric_values_float = torch.where(values, torch.tensor(1.0, device=dev, dtype=torch.float32), torch.tensor(-1.0, device=dev, dtype=torch.float32))
			weight_eff_float = torch.sparse_coo_tensor(indices, numeric_values_float, weight.shape, device=dev, dtype=torch.float32).coalesce()
			z_float = torch.sparse.mm(weight_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense weights are int8: +1, -1, or 0.
			z_float = prevActivation.float() @ weight.float().t() # Cast both to float for matmul

		result = z_float
		#if(propData=="activation"): result = z_float >= segmentActivationThreshold	#not required as segmentActivationThreshold = 1
		
		return result

def getEncodedFeatureSize():
	if(useSequentialSANI):
		encodedFeatureSize = EISANINLPcontinuousVarEncodingNumBits
	else:
		if(useTokenEmbedding):
			encodedFeatureSize = sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
		else:
			encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits
	return encodedFeatureSize
	
def encodeTokensInputIDs(self, x):
	input_ids = x
	if(useTokenEmbedding):
		token_type_ids = torch.zeros_like(input_ids)
		with torch.no_grad():
			embeds = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)	 # [batchSize, sequenceLength, embeddingSize]
		embeds = normalize_to_unit_range(embeds)
		embeds = normalize_unit_range_to_0_to_1(embeds)
	else:
		embeds = input_ids	#will apply one-hot encoding directly to integers
	return embeds

if(useTokenEmbedding):
	def normalize_to_unit_range(embeds: torch.Tensor, per_token: bool = True, eps: float = 1e-12,) -> torch.Tensor:
		"""
		Scale each embedding vector so every element is in [-1, +1].

		Parameters
		----------
		embeds : FloatTensor  [B, L, H]
			Output of `model.embeddings(...)` (or any float tensor).
		per_token : bool, default True
			* True  -> scale each token independently  
			* False -> scale across the entire batch (rarely useful)
		eps : float
			Tiny constant to avoid division by zero.

		Returns
		-------
		norm : FloatTensor  [B, L, H]
			Same shape, values guaranteed in [-1, +1].
		"""
		if per_token:
			# max over hidden dim only \u2192 each token handled separately
			max_abs = embeds.abs().amax(dim=-1, keepdim=True)
		else:
			# max over every element in the tensor
			max_abs = embeds.abs().amax()

		norm = embeds / (max_abs + eps)
		norm = norm.clamp_(-1.0, 1.0)	  # numerical safety (optional)
		return norm

	def normalize_unit_range_to_0_to_1(x: torch.Tensor) -> torch.Tensor:
		"""
		Affine rescale from [-1, +1] -> [0, 1].

		Parameters
		----------
		x : FloatTensor
			Any tensor whose values are already in the closed interval [-1, +1].

		Returns
		-------
		y : FloatTensor
			Same shape as `x`, but with every element mapped so:
				-1 \u2192 0
				 0 \u2192 0.5
				+1 \u2192 1
		"""
		y = (x + 1.0) * 0.5		# (x + 1) / 2
		y = y.clamp_(0.0, 1.0)  # numerical safety (optional)
		return y

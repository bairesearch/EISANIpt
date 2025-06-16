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


	def _sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, initActivation):
	
		device = initActivation.device
		currentActivationTime = calculateTime(batchIndex)
		
		#shift init layer activations (by number input bits) - always place new initActivation units at start of activation/time tensors;
		shiftUnits = windowSize = self.encodedFeatureSize
		self.layerActivations[0] = torch.cat((torch.zeros_like(self.layerActivations[0][..., :shiftUnits]), initActivation), dim=-1)
		self.layerActivations[0][:, :shiftUnits] = initActivation
		self.lastActivationTime[0][:, :shiftUnits] = currentActivationTime
		
		for hiddenLayerIdx in range(self.config.numberOfHiddenLayers):
			layerIdx = hiddenLayerIdx+1
			
			layerActivated = torch.zeros(self.config.batchSize, self.config.hiddenLayerSize, dtype=torch.bool, device=device)

			#segment 1 activations;
			layerSegment0Activated = _compute_layer_sequentialSANI(self, batchIndex, hiddenLayerIdx, SANIsegmentIndexProximal, currentActivationTime, initActivation, device)
			
			#segment 2 activations;
			maxActivationTimeSegment0 = currentActivationTime
			minActivationTimeSegment1 = max(0, currentActivationTime-maxActivationRecallTime)
			for timeIndex in range(maxActivationTimeSegment0, minActivationTimeSegment1):
				layerSegment2ActivatedTimeIndex = _compute_layer_sequentialSANI(self, batchIndex, hiddenLayerIdx, SANIsegmentIndexDistal, timeIndex, initActivation, device)
				layerActivatedTimeIndex = torch.logical_and(layerSegment1ActivatedTimeIndex, layerSegment0Activated)
				layerActivated = torch.logical_and(layerActivated, layerActivatedTimeIndex)

			#update neuron activations
			layerActivatedNot = torch.logical_not(layerActivated)
			self.layerActivations[layerIdx] = torch.logical_or(self.layerActivations[layerIdx], layerActivated)
			self.lastActivationTime[layerIdx] = self.lastActivationTime[layerIdx]*layerActivatedNot.int()	#reset the time values for current neuron activations
			self.lastActivationTime[layerIdx] = self.lastActivationTime[layerIdx]*(layerActivated.int()*currentActivationTime)

			if(trainOrTest and useDynamicGeneratedHiddenConnections):
				timeSeg0 = currentActivationTime
				timeSeg1 = currentActivationTime-layerIdx	#assume network diverges by 1 unit every layer
				prevlayerIdx = layerIdx-1
				prevActivationSeg0 = maskLayerActivationsByTime(self, batchIndex, prevlayerIdx, timeSeg0)
				prevActivationSeg1 = maskLayerActivationsByTime(self, batchIndex, prevlayerIdx, timeSeg1)
				sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx, prevActivationSeg0, prevActivationSeg1, device)

		layerActivationsList = list(self.layerActivations)
		return layerActivationsList

	def calculateTime(batchIndex):
		currentTime = batchIndex
		return currentTime

	def maskLayerActivationsByTime(self, batchIndex, layerIdx, timeIndex):
		if(layerIdx == 0):
			#select subset of layerActivations at timeIndex (of windowSize)
			currentActivationTime = calculateTime(batchIndex)
			timeOffset = currentActivationTime-timeIndex
			windowSize = self.encodedFeatureSize
			activations = self.layerActivations[layerIdx][:, timeOffset:timeOffset+windowSize]
			time = self.lastActivationTime[layerIdx][:, timeOffset:timeOffset+windowSize]
		else:
			activations = self.layerActivations[layerIdx]
			time = self.lastActivationTime[layerIdx]
		timeMask = (time == timeIndex).float()
		maskedActivations = activations*timeMask	#filter activations for same time (at timeIndex)
		return maskedActivations
		
	#returns binary activations
	def _compute_layer_sequentialSANI(self, batchIndex: int, hiddenLayerIdx: int, segmentIdx: int, timeIndex: int, initActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:	
		layerIdx = hiddenLayerIdx+1
		prevlayerIdx = layerIdx-1
		prevActivation = maskLayerActivationsByTime(self, batchIndex, prevlayerIdx, timeIndex)

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

		activated = _segmentActivationFunction(self, z_float, hiddenLayerIdx, segmentIdx)
		return activated

	def _segmentActivationFunction(self, z_float, hiddenLayerIdx, segmentIdx):
		if(sequentialSANIsegmentsConnectedToMultipleLayers):
			segmentActivationThresholdDynamic = int(layerIdx*segmentActivationFractionThreshold)	#round down
			#...
		else:
			activated = z_float >= segmentActivationThreshold
		return activated


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

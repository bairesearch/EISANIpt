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

if(not useNLPcharacterInput):
	model = AutoModel.from_pretrained(bertModelName).eval().to(device)   # embeddingSize-dim vectors

if(useSequentialSANI):

	def _sequentialSANI_dynamic_hidden_growth(self, batchIndex, initActivation):
		#only generate neurons for based on currently activated (activationTime=now) neurons in each layer, and previously activated neurons time contiguous with these
		#TODO
		
	
	def _sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, initActivation):
		for layerIdx in range(self.config.numberOfHiddenLayers):

			layerActivated = torch.zeros(config.hiddenLayerSize, dtype=torch.bool, device=device)

			#segment 1 activations;
			currentActivationTime = calculateTime(batchIndex)
			layerSegment0Activated = _compute_layer_sequentialSANI(self, layerIdx, SANIsegmentIndexProximal, currentActivationTime, initActivation, device)

			#segment 2 activations;
			maxActivationTimeSegment0 = currentActivationTime
			minActivationTimeSegment1 = max(0, currentActivationTime-maxActivationRecallTime)
			for timeIndex in range(minActivationTimeSegment0, minActivationTimeSegment1):
				layerSegment2ActivatedTimeIndex = _compute_layer_sequentialSANI(self, layerIdx, SANIsegmentIndexDistal, timeIndex, initActivation, device)
				layerActivatedTimeIndex = torch.logical_and(layerSegment1ActivatedTimeIndex, layerSegment0Activated)
				layerActivated = torch.logical_and(layerActivated, layerActivatedTimeIndex)

			#update neuron activations
			layerActivatedNot = torch.logical_not(layerActivated)
			self.layerActivations[layerIdx] = torch.logical_or(self.layerActivations[layerIdx], layerActivated)
			self.lastActivationTime[layerIdx] = self.lastActivationTime[layerIdx]*layerActivatedNot.int()	#reset the time values for current neuron activations
			self.lastActivationTime[layerIdx] = self.lastActivationTime[layerIdx]*(layerActivated.int()*currentActivationTime)

			if(trainOrTest and useDynamicGeneratedHiddenConnections):
				EISANIpt_EISANImodelNLP._sequentialSANI_dynamic_hidden_growth(self, batchIndex, layerIdx, initActivation)
					
		layerActivationsList = list(self.layerActivations)
		return layerActivationsList

	def calculateTime(batchIndex):
		currentTime = batchIndex
		return currentTime

	#returns binary activations
	def _compute_layer_sequentialSANI(self, layerIdx: int, segmentIdx: int, timeIndex: int, initActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:	
		if(layerIdx == 0):
			prevActivation = initActivation
		else:	
			#first hidden layer recieves input from input layer (input layer does not have activation times as it is completely activated for every token)
			prevlayerIdx = layerIdx-1
			prevActivation = self.layerActivations[layerIdx, segmentIdx]
			timeMask = (self.lastActivationTime[layerIdx, segmentIdx] == timeIndex).float()
			prevActivation = prevActivation*timeMask	#filter previous activation for same time (at timeIndex)

		# prevActivation is torch.int8 (0 or 1)
		weight = self.hiddenConnectionMatrix[layerIdx][segmentIdx].to(device)

		dev	= prevActivation.device
		# weight = self.hiddenConnectionMatrix[layerIdx].to(dev) # Already done above

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

		activated = self._segmentActivationFunction(layerIdx, segmentIdx, z_float)
		return activated

	def _segmentActivationFunction(self, layerIdx, segmentIdx, z_float):
		segmentActivationThreshold = int((layerIdx+1)*segmentActivationFractionThreshold)	#round down
		self.layerSegmentActivations[layerIdx][segmentIdx] = z_float > segmentActivationThreshold
		return self.layerSegmentActivations[layerIdx][segmentIdx]


def getEncodedFeatureSize():
	if(useNLPcharacterInput):
		encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits
	else:
		encodedFeatureSize = sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
	return encodedFeatureSize
	
def encodeTokensInputIDs(self, x):
	input_ids = x
	if(useNLPcharacterInput):
		embeds = input_ids	#will apply one-hot encoding directly to integers
	else:
		token_type_ids = torch.zeros_like(input_ids)
		with torch.no_grad():
			embeds = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)	 # [batchSize, sequenceLength, embeddingSize]
		embeds = normalize_to_unit_range(embeds)
		embeds = normalize_unit_range_to_0_to_1(embeds)
	return embeds

if(not useNLPcharacterInput):
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

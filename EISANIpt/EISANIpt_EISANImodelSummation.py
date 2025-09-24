"""EISANIpt_EISANImodelSummation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model Summation (summation activated neuronal input)

The EISANI summation activated neuronal input algorithm differs from the original SANI (sequentially activated neuronal input) specification in two ways;
a) tabular/image datasets use summation activated neuronal input. A sequentially activated neuronal input requirement is not enforced, as this was designed for sequential data such as NLP (text).
b) both excitatory and inhibitory input are used (either !useEIneurons:excitatory/inihibitory synapses or useEIneurons:excitatory/inhibitory neurons). 
The algorithm is equivalent to the original SANI specification otherwise (dynamic network generation etc).

Implementation note:
If useEIneurons=False: - a relU function is applied to every hidden neuron (so their output will always be 0 or +1), but connection weights to the next layer can either be positive (+1) or negative (-1).
If useEIneurons=True: - a relU function is applied to both E and I neurons (so their output will always be 0 or +1), but I neurons output is multiplied by -1 before being propagated to next layer, and connection weights are always positive (+1).

"""

import torch
from typing import List, Optional, Tuple
from ANNpt_globalDefs import *
if(useDynamicGeneratedHiddenConnections):
	import EISANIpt_EISANImodelSummationDynamic

if(useConnectionWeights):
	def init_layers(self, config, prevSize):
		# -----------------------------
		# Hidden connection matrices
		# -----------------------------
	
		self.hiddenConnectionMatrix: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueHiddenLayers)]
		self.hiddenConnectionMatrixExcitatory: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueHiddenLayers)]
		self.hiddenConnectionMatrixInhibitory: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueHiddenLayers)]

		for hiddenLayerIdx in range(self.numberUniqueHiddenLayers): # Modified
			for segmentIdx in range(numberOfSegmentsPerNeuron):
				if useEIneurons: # Corrected: use useEIneurons
					if useInhibition:
						excitSize = config.hiddenLayerSize // 2
						inhibSize = config.hiddenLayerSize - excitSize
					else:
						excitSize = config.hiddenLayerSize  # All neurons are excitatory
						inhibSize = 0
					excMat = initialise_layer_weights(self, excitSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
					self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx].append(excMat)
					if useInhibition and inhibSize > 0:
						inhMat = initialise_layer_weights(self, inhibSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
						self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(inhMat)
					else:
						# Create empty inhibitory matrix when inhibition is disabled
						self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(torch.empty(0, prevSize, device=device))
				else:
					mat = initialise_layer_weights(self, config.hiddenLayerSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
					self.hiddenConnectionMatrix[hiddenLayerIdx].append(mat)
			prevSize = config.hiddenLayerSize

	def initialise_layer_weights(self, numNeurons: int, prevSize: int, hiddenLayerIdx: int,) -> torch.Tensor:
		"""
		Create the hidden connection matrix for a single layer.

		- If `useDynamicGeneratedHiddenConnections` is True  -> start empty
		- Else													-> randomly connect
			exactly `self.cfg.numberOfSynapsesPerSegment` synapses
			per neuron.

		Weight values:
			- EI-mode  ->  +1  (excitatory) or always-on +1 (inhibitory handled elsewhere)
			- Standard ->  1 with equal probability
			If sparse and bool: True for +1, False for -1 (non-EI) or True for +1 (EI)
			If dense and int8: +1, -1, or 0.
		"""
		cfg = self.config
		k = cfg.numberOfSynapsesPerSegment  # shorthand
		nnz  = numNeurons * k

		if useSparseHiddenMatrix:

			sparse_dtype = torch.bool

			if useDynamicGeneratedHiddenConnections:
				# start EMPTY
				indices = torch.empty((2, 0), dtype=torch.int64, device=device)
				values  = torch.empty((0,),  dtype=sparse_dtype, device=device)
			else:
				# start with k random synapses per neuron
				'''
				row_idx = torch.arange(numNeurons, device=device).repeat_interleave(k)
				col_idx = torch.randint(prevSize, (numNeurons * k,), device=device)
				indices = torch.stack([row_idx, col_idx])  # shape [2, nnz]
				'''
				# ---------- memory-tight random initialisation ----------
				indices = torch.empty((2, nnz), dtype=torch.int64, device=device)
				# 1. columns: generate in-place
				torch.randint(prevSize, (nnz,), dtype=torch.int64, device=device, out=indices[1])
				# 2. rows: broadcast one small vector into the big slot buffer
				rows = torch.arange(numNeurons, dtype=torch.int64, device=device)  # length = numNeurons
				indices[0].view(numNeurons, k).copy_(rows.unsqueeze(1))		 # expand, copy once

				if useEIneurons:
					values = torch.ones(nnz, device=device, dtype=sparse_dtype)
				else:
					values = torch.randint(0, 2, (nnz,), device=device, dtype=sparse_dtype)

			mat = torch.sparse_coo_tensor(indices, values, size=(numNeurons, prevSize), device=device, dtype=sparse_dtype,).coalesce()
			mat = mat.to(device)
			return mat
		else:
			# -------------------------------------------------- dense initialisation
			weight = torch.zeros(numNeurons, prevSize, device=device, dtype=torch.int8) # Use torch.int8 for dense

			if not useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell
				for n in range(numNeurons):
					syn_idx = torch.randperm(prevSize, device=device)[:k]

					if useEIneurons:
						weight[n, syn_idx] = 1
					else:
						# Generate +1 or -1 with equal probability, as int8
						rand_signs_bool = torch.randint(0, 2, (k,), device=device, dtype=torch.bool)
						rand_signs = torch.where(rand_signs_bool, torch.tensor(1, device=device, dtype=torch.int8), torch.tensor(-1, device=device, dtype=torch.int8))
						weight[n, syn_idx] = rand_signs

			# make it a learnable parameter for the dense case
			return weight
			
# -----------------------------
# Hidden layers
# -----------------------------
		
def summationSANIpassHiddenLayers(self, trainOrTest, initActivation):
	prevActivation = initActivation
	layerActivations: List[torch.Tensor] = []
	for layerIdSuperblock in range(recursiveSuperblocksNumber): # Modified
		for layerIdHidden in range(self.config.numberOfHiddenLayers): # Modified
			uniqueLayerIndex = self._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden)
			if useEIneurons:
				aExc, aInh = compute_layer_EI(self, uniqueLayerIndex, prevActivation, device)
				currentActivation = torch.cat([aExc, aInh], dim=1)
			else:
				currentActivation = compute_layer_standard(self, uniqueLayerIndex, prevActivation, device) # Modified
			
			if(limitConnections and limitHiddenConnections and not useStochasticUpdates):
				self.hiddenNeuronUsage[uniqueLayerIndex] = self.hiddenNeuronUsage[uniqueLayerIndex] + currentActivation.sum(dim=0)	#sum across batch dim
			
			# -------------------------
			# Dynamic hidden connection growth (disabled when stochastic updates enabled)
			# -------------------------
			if (trainOrTest and useDynamicGeneratedHiddenConnections and not useStochasticUpdates):
				for _ in range(numberNeuronSegmentsGeneratedPerSample):
					if(useDynamicGeneratedHiddenConnectionsVectorised):
						EISANIpt_EISANImodelSummationDynamic.dynamic_hidden_growth_vectorised(self, uniqueLayerIndex, prevActivation, currentActivation, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified
					else:
						for s_batch_idx in range(prevActivation.size(0)):                # loop over batch
							prevAct_b  = prevActivation[s_batch_idx : s_batch_idx + 1]             # keep 2- [1, prevSize]
							currAct_b  = currentActivation[s_batch_idx : s_batch_idx + 1]          # keep 2- [1, layerSize]
							EISANIpt_EISANImodelSummationDynamic.dynamic_hidden_growth(self, uniqueLayerIndex, prevAct_b, currAct_b, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified

			layerActivations.append(currentActivation)
			prevActivation = currentActivation
	return layerActivations

def compute_layer_standard(self, hiddenLayerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:
	activatedAllSegments = []

	for segmentIdx in range(numberOfSegmentsPerNeuron):
		# prevActivation is torch.int8 (0 or 1)
		weight = self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIdx].to(device)

		dev	= prevActivation.device
		# weight = self.hiddenConnectionMatrix[hiddenLayerIdx].to(dev) # Already done above

		if useSparseHiddenMatrix:
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
		activated = segmentActivationFunction(self, z_float).to(torch.int8) # bool to int8 (0 or 1)
		activatedAllSegments.append(activated)
	activated = neuronActivationFunction(self, activatedAllSegments)
	return activated

def compute_layer_EI(self, hiddenLayerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
	aExcAllSegments = []
	aInhAllSegments = []
	for segmentIdx in range(numberOfSegmentsPerNeuron):
		# prevActivation is torch.int8 (0 or 1)
		dev  = prevActivation.device
		wExc = self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIdx].to(dev)
		wInh = self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIdx].to(dev)

		# Excitatory
		if useSparseHiddenMatrix:
			# EI sparse weights are True for +1
			numeric_values_exc_float = wExc.values().to(torch.float32) # True becomes 1.0
			wExc_eff_float = torch.sparse_coo_tensor(wExc.indices(), numeric_values_exc_float, wExc.shape, device=dev, dtype=torch.float32).coalesce()
			zExc_float = torch.sparse.mm(wExc_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense EI weights are 1 (int8). Convert to float for matmul.
			zExc_float = prevActivation.float() @ wExc.float().t()
		aExc = segmentActivationFunction(self, zExc_float).to(torch.int8) # bool to int8 (0 or 1)
		aExcAllSegments.append(aExc)

		# Inhibitory
		if useSparseHiddenMatrix:
			# EI sparse weights are True for +1
			numeric_values_inh_float = wInh.values().to(torch.float32) # True becomes 1.0
			wInh_eff_float = torch.sparse_coo_tensor(wInh.indices(), numeric_values_inh_float, wInh.shape, device=dev, dtype=torch.float32).coalesce()
			zInh_float = torch.sparse.mm(wInh_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense EI weights are 1 (int8). Convert to float for matmul.
			zInh_float = prevActivation.float() @ wInh.float().t()
		firesInh = segmentActivationFunction(self, zInh_float)
		aInh = torch.zeros_like(zInh_float, dtype=torch.int8, device=dev) # Initialize with correct shape, device and int8 type
		aInh[firesInh] = -1
		aInhAllSegments.append(aInh)
	aExc = neuronActivationFunction(self, aExcAllSegments)
	aInh = neuronActivationFunction(self, aInhAllSegments)
	return aExc, aInh

def segmentActivationFunction(self, z_all_segments):
    # When using stochastic updates, treat any positive net input as a fire.
    if useStochasticUpdates:
        segment_fires = z_all_segments > segmentActivationThreshold
    else:
        segment_fires = z_all_segments >= segmentActivationThreshold
    return segment_fires

def neuronActivationFunction(self, a_all_segments_list):
	#for EISANI summation activated neuronal input assume neuron is activated when any segments are activated
	a_all_segments = torch.stack(a_all_segments_list, dim=2)	# [B, numNeurons, numberOfSegmentsPerNeuron] (bool)
	# Combine segment activations: a neuron is active if any of its segments are active.
	neuron_fires = torch.any(a_all_segments, dim=2) # [B, numNeurons] (bool)
	return neuron_fires

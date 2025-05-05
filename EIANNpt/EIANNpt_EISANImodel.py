"""EIANNpt_EISANImodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EIANNpt excitatory inhibitory (EI) sequentially activated neuronal input (SANI) network model

The sequentially activated neuronal input requirement is not enforced for tabular input, but the algorithm is equivalent to SANI otherwise.

Implementation note:
If useEIneurons=False: - a relU function is applied to every hidden neuron (so their output will always be 0 or +1), but connection weights to the next layer can either be positive (+1) or negative (-1).
If useEIneurons=True: - a relU function is applied to both E and I neurons (so their output will always be 0 or +1), but I neurons output is multiplied by -1 before being propagated to next layer.

"""

import torch
from torch import nn
from ANNpt_globalDefs import *
#from torchmetrics.classification import Accuracy
from typing import List, Optional, Tuple

class EISANIconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, numberOfFeatures, numberOfClasses, numberOfSynapsesPerSegment):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.numberOfSynapsesPerSegment = numberOfSynapsesPerSegment
		self.numberOfHiddenLayers = numberOfLayers - 1

class Loss:
	def __init__(self, value=0.0):
		self._value = value

	def item(self):
		return self._value


# -------------------------------------------------------------
# Encoding helpers
# -------------------------------------------------------------

def _continuous_to_int(x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""Map (batch, features) continuous tensor to integer levels [0, 2**bits-1]."""
	xClamped = torch.clamp(x, minVal, maxVal)
	norm = (xClamped - minVal) / (maxVal - minVal)
	scaled = norm * (2 ** numBits - 1)
	return scaled.round().to(torch.long)


def gray_code_encode(x: torch.Tensor, numBits: int, minVal: float, maxVal: float,) -> torch.Tensor:
	"""Vectorised Gray-code encoding (batch, features, bits)."""
	intLevels = _continuous_to_int(x, numBits, minVal, maxVal)
	grayLevels = intLevels ^ (intLevels >> 1)
	bitPositions = torch.arange(numBits, device=x.device)
	bits = ((grayLevels.unsqueeze(-1) >> bitPositions) & 1).float()
	return bits  # (batch, features, bits)


def thermometer_encode(x: torch.Tensor, numBits: int, minVal: float, maxVal: float,) -> torch.Tensor:
	"""Vectorised thermometer encoding (batch, features, bits)."""
	intLevels = _continuous_to_int(x, numBits, minVal, maxVal)
	thresholds = torch.arange(numBits, device=x.device)
	bits = (intLevels.unsqueeze(-1) >= thresholds).float()
	return bits


# -------------------------------------------------------------
# Core network module
# -------------------------------------------------------------

class EISANImodel(nn.Module):
	"""Custom binary neural network implementing the EISANI specification."""

	def __init__(self, config: EISANIconfig) -> None:
		super().__init__()

		# -----------------------------
		# Public config
		# -----------------------------
		self.config = config
		self.useSparseMatrix = useSparseMatrix
		self.useEIneurons = useEIneurons
		self.useDynamicGeneratedHiddenConnections = useDynamicGeneratedHiddenConnections
		self.useBinaryOutputConnections = useBinaryOutputConnections
		self.useGrayCode = useGrayCode
		self.continuousVarMin = continuousVarMin
		self.continuousVarMax = continuousVarMax
		self.continuousVarEncodingNumBits = continuousVarEncodingNumBits
		self.segmentActivationThreshold = segmentActivationThreshold
		self.targetActivationSparsityFraction = targetActivationSparsityFraction

		# -----------------------------
		# Derived sizes
		# -----------------------------
		self.encodedFeatureSize = config.numberOfFeatures * continuousVarEncodingNumBits
		prevSize = self.encodedFeatureSize

		# -----------------------------
		# Hidden connection matrices
		# -----------------------------
		self.hiddenConnectionMatrix: List[torch.Tensor] = []
		self.hiddenConnectionMatrixExcitatory: List[torch.Tensor] = []
		self.hiddenConnectionMatrixInhibitory: List[torch.Tensor] = []

		for layerIdx in range(config.numberOfHiddenLayers):
			if useEIneurons:
				excitSize = config.hiddenLayerSize // 2
				inhibSize = config.hiddenLayerSize - excitSize
				excMat = self._initialise_layer_weights(excitSize, prevSize, useSparseMatrix)
				inhMat = self._initialise_layer_weights(inhibSize, prevSize, useSparseMatrix)
				self.hiddenConnectionMatrixExcitatory.append(excMat)
				self.hiddenConnectionMatrixInhibitory.append(inhMat)
			else:
				mat = self._initialise_layer_weights(config.hiddenLayerSize, prevSize, useSparseMatrix)
				self.hiddenConnectionMatrix.append(mat)
			prevSize = config.hiddenLayerSize

		if useDynamicGeneratedHiddenConnections:
			self.register_buffer("neuronSegmentActivatedMask", torch.zeros(config.numberOfHiddenLayers, config.hiddenLayerSize, dtype=torch.bool,),)

		# -----------------------------
		# Output connection matrix
		# -----------------------------
		outConnShape = (config.numberOfHiddenLayers, config.hiddenLayerSize, config.numberOfClasses,)
		if useBinaryOutputConnections:
			self.register_buffer("outputConnectionMatrix", torch.zeros(outConnShape, dtype=torch.bool),)
		else:
			self.register_buffer("outputConnectionMatrix", torch.zeros(outConnShape, dtype=torch.float),)

	# ---------------------------------------------------------
	# Helper - layer initialisation
	# ---------------------------------------------------------

	def _initialise_layer_weights(self, numNeurons: int, prevSize: int, layerIdx: int,) -> torch.Tensor:
		"""
		Create the hidden connection matrix for a single layer.

		- If `self.useDynamicGeneratedHiddenConnections` is True  -> start empty
		- Else													-> randomly connect
		  exactly `self.cfg.numberOfSynapsesPerSegment` synapses
		  per neuron.

		Weight values:
		  - EI-mode  ->  +1  (excitatory) or always-on +1 (inhibitory handled elsewhere)
		  - Standard ->  1 with equal probability
		"""
		cfg = self.config
		dev = device  # ensures GPU/CPU consistency
		k = cfg.numberOfSynapsesPerSegment  # shorthand

		if self.useSparseMatrix:
			# -------------------------------------------------- sparse initialisation
			if self.useDynamicGeneratedHiddenConnections:
				# start EMPTY
				indices = torch.empty((2, 0), dtype=torch.long, device=dev)
				values  = torch.empty((0,), dtype=torch.float32, device=dev)
			else:
				# start with k random synapses per neuron
				row_idx = torch.arange(numNeurons, device=dev).repeat_interleave(k)
				col_idx = torch.randint(prevSize, (numNeurons * k,), device=dev)
				indices = torch.stack([row_idx, col_idx])  # shape [2, nnz]

				if self.useEIneurons:
					values = torch.ones(indices.size(1), device=dev)
				else:
					# 1 with equal probability
					values = torch.randint(0, 2, (indices.size(1),), device=dev, dtype=torch.float32) * 2 - 1

			mat = torch.sparse_coo_tensor(indices, values, size=(numNeurons, prevSize), device=dev, dtype=torch.float32,).coalesce()
			return mat
		else:
			# -------------------------------------------------- dense initialisation
			weight = torch.zeros(numNeurons, prevSize, device=dev)

			if not self.useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell
				for n in range(numNeurons):
					syn_idx = torch.randperm(prevSize, device=dev)[:k]

					if self.useEIneurons:
						weight[n, syn_idx] = 1.0
					else:
						rand_signs = torch.randint(0, 2, (k,), device=dev, dtype=torch.float32) * 2 - 1
						weight[n, syn_idx] = rand_signs

			# make it a learnable parameter for the dense case
			return nn.Parameter(weight)


	# ---------------------------------------------------------
	# Forward pass
	# ---------------------------------------------------------

	def forward(self, trainOrTest: bool, x: torch.Tensor, y: Optional[torch.Tensor] = None, optim=None, l=None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass.
		
		Args:
			trainOrTest: True=> train mode; False=> inference.
			x: (batch, features) tensor in continuous range.
			y: (batch,) int64 labels when trainOrTest==True.

		Returns:
			predictions, outputActivations (both shape (batch, classes)).
		"""
		batchSize = x.size(0)
		assert (batchSize == self.config.batchSize), "Batch size must match config.batchSize"
		device = x.device

		# -----------------------------
		# Encode inputs
		# -----------------------------
		if self.useGrayCode:
			encoded = gray_code_encode(x, self.continuousVarEncodingNumBits, self.continuousVarMin, self.continuousVarMax,)
		else:
			encoded = thermometer_encode(x, self.continuousVarEncodingNumBits, self.continuousVarMin, self.continuousVarMax,)
		# Flatten feature & bit dimensions -> (batch, encodedFeatureSize)
		prevActivation = encoded.view(batchSize, -1)
		
		if(debugEISANIoutput):
			print("Input bits that are 1 :", int(prevActivation.sum().item()))
		
		# -----------------------------
		# Pass through hidden layers
		# -----------------------------
		layerActivations: List[torch.Tensor] = []

		for layerIdx in range(self.config.numberOfHiddenLayers):
			if self.useEIneurons:
				aExc, aInh = self._compute_layer_EI(layerIdx, prevActivation, device)
				currentActivation = torch.cat([aExc, aInh], dim=1)
			else:
				currentActivation = self._compute_layer_standard(layerIdx, prevActivation, device)

			layerActivations.append(currentActivation)

			# -------------------------
			# Dynamic hidden connection growth
			# -------------------------
			if (trainOrTest and self.useDynamicGeneratedHiddenConnections):
				self._dynamic_hidden_growth(layerIdx, prevActivation, currentActivation, device)

			prevActivation = currentActivation

		# -----------------------------
		# Output layer
		# -----------------------------
		outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

		for layerIdx, act in enumerate(layerActivations):
			if self.useBinaryOutputConnections:
				weights = self.outputConnectionMatrix[layerIdx].float()
			else:
				weights = self.outputConnectionMatrix[layerIdx]
			outputActivations += act @ weights

			# Training: reinforce output connections
			if trainOrTest and y is not None:
				self._update_output_connections(layerIdx, act, y, device)

		predictions = torch.argmax(outputActivations, dim=1)
		
		# count how many are exactly correct
		correct = (predictions == y).sum().item()
		accuracy = correct / y.size(0)
		loss = Loss(0.0)
		
		return loss, accuracy

	# ---------------------------------------------------------
	# Layer computation helpers
	# ---------------------------------------------------------

	def _compute_layer_standard(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:
		weight = self.hiddenConnectionMatrix[layerIdx]
		
		if(debugEISANIoutput):
			layer0 = self.hiddenConnectionMatrix[0]
			if(useSparseMatrix):
				print("Non-zero synapses in layer-0:", layer0._nnz())
			else:
				print("Non-zero synapses in layer-0:", layer0.count_nonzero())

		# ensure our hidden-connection sparse tensor is on the same device as prevActivation
		dev	= prevActivation.device
		weight = self.hiddenConnectionMatrix[layerIdx].to(dev)

		if weight.is_sparse:
			z = torch.sparse.mm(weight, prevActivation.t()).t()
		else:
			z = prevActivation @ weight.t()
		
		if(debugEISANIoutput):
			#print("z = ", z)
			# z is [batch, layerSize]
			print("Layer-0 z stats: min", z.min().item(), "max", z.max().item(), "mean", z.float().mean().item())
		  
		activated = (z >= self.segmentActivationThreshold).float()
		return activated

	def _compute_layer_EI(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
		dev  = prevActivation.device
		wExc = self.hiddenConnectionMatrixExcitatory[layerIdx].to(dev)
		wInh = self.hiddenConnectionMatrixInhibitory[layerIdx].to(dev)
		# Excitatory
		if wExc.is_sparse:
			zExc = torch.sparse.mm(wExc, prevActivation.t()).t()
		else:
			zExc = prevActivation @ wExc.t()
		aExc = (zExc >= self.segmentActivationThreshold).float()
		# Inhibitory
		if wInh.is_sparse:
			zInh = torch.sparse.mm(wInh, prevActivation.t()).t()
		else:
			zInh = prevActivation @ wInh.t()
		firesInh = zInh >= self.segmentActivationThreshold
		aInh = torch.zeros_like(zInh)
		aInh[firesInh] = -1.0
		return aExc, aInh

	# ---------------------------------------------------------
	# Dynamic hidden growth helper
	# ---------------------------------------------------------

	def _dynamic_hidden_growth(self, layerIdx: int, prevActivation: torch.Tensor, currentActivation: torch.Tensor, device: torch.device,) -> None:
		batchActiveMask = currentActivation != 0.0
		fractionActive = batchActiveMask.float().mean().item()
		if(debugEISANIoutput):
			print("fractionActive = ", fractionActive)
		if fractionActive >= self.targetActivationSparsityFraction:
			return  # sparsity satisfied

		# Need to activate a new neuron (one per call)
		available = (~self.neuronSegmentActivatedMask[layerIdx]).nonzero(as_tuple=True)[0]
		if available.numel() == 0:
			if self.training:
				print(f"Warning: no more available neurons in hidden layer {layerIdx}")
			return
		newNeuronIdx = available[0].item()
		self.neuronSegmentActivatedMask[layerIdx, newNeuronIdx] = True


		"""
		# -------------------------------------------------------
		implementation 1a: Sample synapses - random formation of synapses to previous layer neurons;
		# -------------------------------------------------------
		# Sample synapses
		numSyn = self.config.numberOfSynapsesPerSegment
		prevSize = prevActivation.size(1)
		randIdx = torch.randperm(prevSize, device=device)[:numSyn]
		prevActSample = prevActivation[0, randIdx]  # use first sample as reference
		weights = torch.where(prevActSample > 0, torch.ones(numSyn, device=device), -torch.ones(numSyn, device=device))
		"""
		
		"""
		# -------------------------------------------------------
		# implementation 1b: Sample synapses - favour *currently active* presynaptic neurons
		# -------------------------------------------------------
		numSyn   = self.config.numberOfSynapsesPerSegment
		presyn   = prevActivation[0]						 # [prevSize] 0./1.

		activeIdx   = (presyn > 0).nonzero(as_tuple=True)[0]		# on-bits
		inactiveIdx = (presyn == 0).nonzero(as_tuple=True)[0]	   # off-bits

		if activeIdx.numel() >= numSyn:
			# enough active; pick k of them at random
			choose = torch.randperm(activeIdx.numel(), device=device)[:numSyn]
			randIdx = activeIdx[choose]
		else:
			# take all active, then fill the remainder with random inactive
			need	 = numSyn - activeIdx.numel()
			fill_idx = torch.randperm(inactiveIdx.numel(), device=device)[:need]
			randIdx  = torch.cat([activeIdx, inactiveIdx[fill_idx]], dim=0)
		"""

		# -------------------------------------------------------
		# implementation 1c: Sample synapses - 50% active, 50% inactive
		# - 50% of synapses to be connected to randomly selected previous layer active neurons, and b) 50% of synapses to be connected to randomly selected previous layer inactive neurons (or if useEIneurons: randomly selected inactive previous layer inhibitory neurons).
		# - midway between implementation 1a and implementation 1b
		# -------------------------------------------------------
		numSyn   = self.config.numberOfSynapsesPerSegment
		#presyn   = prevActivation[0]						 # [prevSize] 0./1.	#orig
		presyn   = (prevActivation > 0).any(dim=0).float()   # [prevSize] 0./1.
		
		activeIdx   = (presyn > 0).nonzero(as_tuple=True)[0]		  # on-bits

		if self.useEIneurons:
			half		= self.config.hiddenLayerSize // 2			# split E/I
			inh_mask	= torch.arange(presyn.numel(), device=device) >= half
			inactiveIdx = ((presyn == 0) & inh_mask).nonzero(as_tuple=True)[0]
		else:
			inactiveIdx = (presyn == 0).nonzero(as_tuple=True)[0]	 # off-bits

		# --- decide how many active / inactive synapses we need
		if useEIneurons:
			numActive   = numSyn // 2	#orig
		else:
			numActive   = (numSyn + 1) // 2        # ceiling - bias positive
		numInactive = numSyn - numActive

		# guard against edge cases where pool size is smaller than desired
		numActive   = min(numActive,   activeIdx.numel())
		numInactive = min(numInactive, inactiveIdx.numel())
		shortfall   = numSyn - (numActive + numInactive)

		# if one pool is too small, take the remainder from the other
		if shortfall > 0:
			if activeIdx.numel() - numActive >= shortfall:
				numActive += shortfall
			else:
				numInactive += shortfall

		chooseA = torch.randperm(activeIdx.numel(),   device=device)[:numActive]
		chooseI = torch.randperm(inactiveIdx.numel(), device=device)[:numInactive]

		randIdx = torch.cat([activeIdx[chooseA], inactiveIdx[chooseI]], dim=0)

		# permute so the order is still random
		randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]



		# weights: +1 if presynaptic *currently on*, else -1
		prevActSample = presyn[randIdx]						# 0./1.
		#weights = torch.where(prevActSample > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))	#orig
		weights = torch.where(presyn[randIdx] > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))

		# Update hidden connection matrix
		if self.useEIneurons:
			# Decide whether the new neuron is excitatory or inhibitory based on index
			half = self.config.hiddenLayerSize // 2
			if newNeuronIdx < half:
				mat = self.hiddenConnectionMatrixExcitatory[layerIdx]
				relativeIdx = newNeuronIdx
			else:
				mat = self.hiddenConnectionMatrixInhibitory[layerIdx]
				relativeIdx = newNeuronIdx - half
		else:
			mat = self.hiddenConnectionMatrix[layerIdx]
			relativeIdx = newNeuronIdx

		if mat.is_sparse:
			# Append to sparse matrix
			existing_indices = mat._indices()
			existing_values = mat._values()

			new_indices = torch.stack([torch.full((numSyn,), relativeIdx, device=device, dtype=torch.long), randIdx,])
			new_values = weights
			
			dev = existing_indices.device
			new_indices = new_indices.to(dev)
			new_values  = new_values.to(dev)
			
			matNew = torch.sparse_coo_tensor(torch.cat([existing_indices, new_indices], dim=1), torch.cat([existing_values, new_values]), mat.size(), device=dev,)
			if self.useEIneurons and newNeuronIdx >= half:
				self.hiddenConnectionMatrixInhibitory[layerIdx] = matNew
			elif self.useEIneurons:
				self.hiddenConnectionMatrixExcitatory[layerIdx] = matNew
			else:
				self.hiddenConnectionMatrix[layerIdx] = matNew
		else:
			# structural update - do NOT track in autograd
			with torch.no_grad():
				mat[relativeIdx, randIdx] = weights

	# ---------------------------------------------------------
	# Output connection update helper
	# ---------------------------------------------------------

	def _update_output_connections(self, layerIdx: int, activation: torch.Tensor, y: torch.Tensor, device: torch.device,) -> None:
		batchSize = activation.size(0)
		activeMask = activation != 0.0  # (batch, hidden)
		for sampleIdx in range(batchSize):
			targetClass = y[sampleIdx].item()
			activeNeurons = activeMask[sampleIdx].nonzero(as_tuple=True)[0]
			if activeNeurons.numel() == 0:
				continue
			if self.useBinaryOutputConnections:
				self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] = True
			else:
				self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] += 1.0

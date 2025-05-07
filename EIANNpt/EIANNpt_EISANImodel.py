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
			self.register_buffer("neuronSegmentAssignedMask", torch.zeros(config.numberOfHiddenLayers, config.hiddenLayerSize, dtype=torch.bool,),)

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
				for _ in range(numberNeuronsGeneratedPerSample):
					if(useDynamicGeneratedHiddenConnectionsVectorised):
						self._dynamic_hidden_growth_vectorised(layerIdx, prevActivation, currentActivation, device)
					else:
						for s in range(prevActivation.size(0)):                # loop over batch
							prevAct_b  = prevActivation[s : s + 1]             # keep 2- [1, prevSize]
							currAct_b  = currentActivation[s : s + 1]          # keep 2- [1, layerSize]
							self._dynamic_hidden_growth(layerIdx, prevAct_b, currAct_b, device)

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
		available = (~self.neuronSegmentAssignedMask[layerIdx]).nonzero(as_tuple=True)[0]
		if available.numel() == 0:
			if self.training:
				print(f"Warning: no more available neurons in hidden layer {layerIdx}")
			return
		newNeuronIdx = available[0].item()
		self.neuronSegmentAssignedMask[layerIdx, newNeuronIdx] = True


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

		numSyn   = self.config.numberOfSynapsesPerSegment
		#presyn   = prevActivation[0]						 # [prevSize] 0./1.	#orig
		presyn   = (prevActivation > 0).any(dim=0).float()   # [prevSize] 0./1.
		
		if self.useEIneurons:
			# -------------------------------------------------------
			# implementation 1d: Sample synapses - 50% active, 50% inactive
			# For useEIneurons=True;
			# - the excitatory neuron segment connections should be defined based on 50% active presynaptic layer excitatory neurons, and 50% inactive presynaptic layer inhibitory neurons.
			# - the inhibitory neuron segment connections should be defined based on 50% inactive presynaptic layer excitatory neurons, and 50% active presynaptic layer inhibitory neurons.
			# -------------------------------------------------------

			# ---------- EI-specific pools ----------
			halfPrev		= presyn.numel() // 2
			isExcPresyn	 = torch.arange(presyn.numel(), device=device) < halfPrev
			isInhPresyn	 = ~isExcPresyn

			excActiveIdx	= (isExcPresyn & (presyn > 0)).nonzero(as_tuple=True)[0]
			excInactiveIdx  = (isExcPresyn & (presyn == 0)).nonzero(as_tuple=True)[0]
			inhActiveIdx	= (isInhPresyn & (presyn > 0)).nonzero(as_tuple=True)[0]
			inhInactiveIdx  = (isInhPresyn & (presyn == 0)).nonzero(as_tuple=True)[0]

			halfThis   = self.config.hiddenLayerSize // 2
			isExcNeur  = newNeuronIdx < halfThis

			if isExcNeur:
				activePool   = excActiveIdx	  # active E
				inactivePool = inhInactiveIdx	# inactive I
			else:
				activePool   = inhActiveIdx	  # active I
				inactivePool = excInactiveIdx	# inactive E

			# --- decide how many active / inactive synapses we need
			if useActiveBias:
				numActive   = (numSyn + 1) // 2		# ceiling - bias positive	# bias for odd k	# allocate counts (ceil for odd k)
			else:
				numActive   = numSyn // 2	#orig
			numInactive = numSyn - numActive

			chooseA = torch.randperm(activePool.numel(),   device=device)[:numActive]
			chooseI = torch.randperm(inactivePool.numel(), device=device)[:numInactive]

			randIdx = torch.cat([activePool[chooseA], inactivePool[chooseI]], dim=0)
			# make sure we have exactly `numSyn` indices
			shortfall = numSyn - randIdx.numel()
			if shortfall > 0:
				# borrow from the other pool (wraps around if needed)
				filler = inactivePool if activePool.numel() >= inactivePool.numel() else activePool
				extra  = filler[torch.randperm(filler.numel(), device=device)[:shortfall]]
				randIdx = torch.cat([randIdx, extra], dim=0)
			randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]  #shuffle - permute so the order is still random

			# EI mode -> all weights are +1
			weights = torch.ones_like(randIdx, dtype=torch.float32, device=device)
		else:
			# -------------------------------------------------------
			# implementation 1c: Sample synapses - 50% active, 50% inactive
			# For useEIneurons=False;
			# - 50% of synapses to be connected to randomly selected previous layer active neurons
			# - 50% of synapses to be connected to randomly selected previous layer inactive neurons
			# - midway between implementation 1a and implementation 1b
			# -------------------------------------------------------

			activeIdx   = (presyn > 0).nonzero(as_tuple=True)[0]		  # on-bits
			inactiveIdx = (presyn == 0).nonzero(as_tuple=True)[0]	 # off-bits

			# --- decide how many active / inactive synapses we need
			if useActiveBias:
				numActive   = (numSyn + 1) // 2		# ceiling - bias positive	# bias for odd k
			else:
				numActive   = numSyn // 2	#orig
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
			# make sure we have exactly `numSyn` indices
			shortfall = numSyn - randIdx.numel()
			if shortfall > 0:
				# borrow from the other pool (wraps around if needed)
				filler = inactivePool if activePool.numel() >= inactivePool.numel() else activePool
				extra  = filler[torch.randperm(filler.numel(), device=device)[:shortfall]]
				randIdx = torch.cat([randIdx, extra], dim=0)
			randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]	#shuffle - permute so the order is still random

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

			new_indices = torch.stack([torch.full((randIdx.numel(),), relativeIdx, device=device, dtype=torch.long), randIdx,])
			new_values = weights
			
			dev = existing_indices.device
			new_indices = new_indices.to(dev)
			new_values  = new_values.to(dev)
			
			matNew = torch.sparse_coo_tensor(torch.cat([existing_indices, new_indices], dim=1), torch.cat([existing_values, new_values]), mat.size(), device=dev,)
		else:
			# structural update - do NOT track in autograd
			with torch.no_grad():
				mat[relativeIdx, randIdx] = weights
			matNew = mat
			
		if self.useEIneurons and newNeuronIdx >= half:
			self.hiddenConnectionMatrixInhibitory[layerIdx] = matNew
		elif self.useEIneurons:
			self.hiddenConnectionMatrixExcitatory[layerIdx] = matNew
		else:
			self.hiddenConnectionMatrix[layerIdx] = matNew

	@torch.no_grad()
	def _dynamic_hidden_growth_vectorised(self, layerIdx: int, prevActivation: torch.Tensor, currActivation: torch.Tensor, device: torch.device,) -> None:
		"""
		Vectorised growth: handles an entire batch, but creates at most ONE new
		neuron per *sample* that is below the sparsity target.  Exactly follows
		the spec for both EI and non-EI modes.
		
		prevActivation/currActivation: [B, prevSize] float {0,1}
		"""
		cfg	= self.config
		k	  = cfg.numberOfSynapsesPerSegment
		B, P   = prevActivation.shape
		Lsize  = currActivation.size(1)

		# ---------- 1. which samples need a neuron? ------------------------------
		fracAct = currActivation.float().mean(dim=1)				 # [B]
		if(debugEISANIoutput):
			print("fracAct = ", fracAct)
		growMask = fracAct < self.targetActivationSparsityFraction   # bool [B]
		if not growMask.any():
			return

		growSamples = growMask.nonzero(as_tuple=False).squeeze(1)	# [G] sample idx
		G = growSamples.numel()

		# ---------- 2. reserve G unused neuron slots -----------------------------
		avail = (~self.neuronSegmentAssignedMask[layerIdx]).nonzero(as_tuple=True)[0]
		if avail.numel() < G:
			G = avail.numel()
			growSamples = growSamples[:G]
			if G == 0:
				if self.training:
					print(f"Warning: no free neurons in layer {layerIdx}")
				return
		newRows = avail[:G]										  # [G]
		self.neuronSegmentAssignedMask[layerIdx, newRows] = True

		# ---------- 3. vectorised synapse sampling -------------------------------
		# presynBatch: [G, P] float {0,1}
		presynBatch = prevActivation[growSamples]
		onMask	  = presynBatch > 0								# bool
		offMask	 = ~onMask

		# -- helper to draw N unique indices from a row mask ----------------------
		def draw_indices(mask: torch.Tensor, n: int) -> torch.Tensor:
			"""
			mask : [G, P]  bool
			returns [G, n] long, allowing repeats if a row has < n True
			"""
			prob = mask.float() + 1e-6							   # avoid zero row
			idx  = torch.multinomial(prob, n, replacement=True)	  # [G, n]
			return idx

		# decide counts
		if getattr(self, "useActiveBias", True):
			nA = (k + 1) // 2	# ceil(k/2)
		else:
			nA = k // 2
		nI = k - nA

		# EI-aware pools -----------------------------------------------------------
		if self.useEIneurons:
			halfPrev   = P // 2
			isExcPre   = torch.arange(P, device=device) < halfPrev   # [P] bool
			isInhPre   = ~isExcPre

			# split masks
			excOn   = onMask  &  isExcPre
			excOff  = offMask &  isExcPre
			inhOn   = onMask  &  isInhPre
			inhOff  = offMask &  isInhPre

			# neuron type for each new row
			halfThis  = cfg.hiddenLayerSize // 2
			isExcNeur = newRows < halfThis						   # [G] bool

			# choose pools per neuron in vectorised form
			activePool   = torch.where(isExcNeur.unsqueeze(1), excOn,  inhOn)
			inactivePool = torch.where(isExcNeur.unsqueeze(1), inhOff, excOff)

			actIdx  = draw_indices(activePool,   nA)				 # [G, nA]
			inIdx   = draw_indices(inactivePool, nI)				 # [G, nI]
			colIdx  = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]
			weights = torch.ones(G, k, device=device)				# all +1
		else:
			# non-EI: 50 / 50 active / inactive
			actIdx  = draw_indices(onMask,  nA)					  # [G, nA]
			inIdx   = draw_indices(offMask, nI)					  # [G, nI]
			colIdx  = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]

			presynPicked = presynBatch.gather(1, colIdx)			 # 0./1., [G,k]
			weights = torch.where(presynPicked > 0, torch.ones_like(presynPicked), -torch.ones_like(presynPicked))	# ±1

		# shuffle cols inside each row to keep random order
		perm = torch.randperm(k, device=device)
		colIdx = colIdx[:, perm]
		weights = weights[:, perm]

		# ---------- 4. flatten to COO lists --------------------------------------
		flatRows = newRows.repeat_interleave(k)					  # [G*k]
		flatCols = colIdx.reshape(-1)								# [G*k]
		flatVals = weights.reshape(-1)							   # [G*k]

		# ---------- 5. write to weight matrix (sparse or dense) ------------------
		if self.useEIneurons:
			half = cfg.hiddenLayerSize // 2
			if (newRows[0] < half):
				targetMats = self.hiddenConnectionMatrixExcitatory
			else:
				targetMats = self.hiddenConnectionMatrixInhibitory
			mat = targetMats[layerIdx]
		else:
			mat = self.hiddenConnectionMatrix[layerIdx]

		if mat.is_sparse:
			idx_new = torch.stack([flatRows, flatCols], dim=0)	   # [2, G*k]
			val_new = flatVals
			mat_new = torch.sparse_coo_tensor(torch.cat([mat.indices(), idx_new], dim=1), torch.cat([mat.values(),  val_new]), size=mat.size(), device=device,).coalesce()
		else:	# dense case
			mat_new = mat.clone()
			mat_new[flatRows, flatCols] = flatVals

		# ---------- 6. store back -------------------------------------------------
		if self.useEIneurons:
			if newRows[0] < half:
				self.hiddenConnectionMatrixExcitatory[layerIdx] = mat_new
			else:
				self.hiddenConnectionMatrixInhibitory[layerIdx] = mat_new
		else:
			self.hiddenConnectionMatrix[layerIdx] = mat_new


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

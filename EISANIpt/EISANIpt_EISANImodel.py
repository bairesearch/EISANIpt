"""EISANIpt_EISANImodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt excitatory inhibitory (EI) sequentially/summation activated neuronal input (SANI) network model

The EISANI algorithm differs from the original SANI (sequentially activated neuronal input) specification in two ways;
a) tabular/image datasets use summation activated neuronal input. A sequentially activated neuronal input requirement is not enforced, as this was designed for sequential data such as NLP (text).
b) both excitatory and inhibitory input are used (either !useEIneurons:excitatory/inihibitory synapses or useEIneurons:excitatory/inhibitory neurons). 
The algorithm is equivalent to the original SANI specification otherwise (dynamic network generation etc).

Implementation note:
If useEIneurons=False: - a relU function is applied to every hidden neuron (so their output will always be 0 or +1), but connection weights to the next layer can either be positive (+1) or negative (-1).
If useEIneurons=True: - a relU function is applied to both E and I neurons (so their output will always be 0 or +1), but I neurons output is multiplied by -1 before being propagated to next layer, and connection weights are always positive (+1).

"""

import torch
from torch import nn
from ANNpt_globalDefs import *
from typing import List, Optional, Tuple
if(useDynamicGeneratedHiddenConnections):
	import EISANIpt_EISANImodelDynamic
if(useImageDataset):
	import EISANIpt_EISANImodelCNN


class EISANIconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, inputLayerSize, outputLayerSize, numberOfFeatures, numberOfClasses, numberOfSynapsesPerSegment, fieldTypeList):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.numberOfConvlayers = numberOfConvlayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.numberOfSynapsesPerSegment = numberOfSynapsesPerSegment
		if(useImageDataset):
			numberOfLinearLayers = numberOfLayers - numberOfConvlayers
			self.numberOfHiddenLayers = numberOfLinearLayers - 1
		else:
			self.numberOfHiddenLayers = numberOfLayers - 1
		self.fieldTypeList = fieldTypeList

class Loss:
	def __init__(self, value=0.0):
		self._value = value

	def item(self):
		return self._value


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
		self.segmentActivationThreshold = segmentActivationThreshold
		self.targetActivationSparsityFraction = targetActivationSparsityFraction
		self.useOutputConnectionsLastLayer = useOutputConnectionsLastLayer

		# -----------------------------
		# Derived sizes
		# -----------------------------
		if useImageDataset:
			EISANIpt_EISANImodelCNN._init_conv_layers(self)                  # fills self.convKernels & self.encodedFeatureSize

			#if(EISANICNNdynamicallyGenerateLinearInputFeatures):	
			#	self.CNNoutputEncodedFeaturesDict = {}	#key: CNNoutputLayerFlatFeatureIndex, value: linearInputLayerFeatureIndex	#non-vectorised implementation
		else:
			# self.encodedFeatureSize = config.numberOfFeatures * continuousVarEncodingNumBits # Old
			fieldTypeList = config.fieldTypeList
			if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
				self.encodedFeatureSize = 0
				for i in range(config.numberOfFeatures):
					if i < len(fieldTypeList) and fieldTypeList[i] == 'bool':
						self.encodedFeatureSize += 1
						#print("bool detected")
					else:
						self.encodedFeatureSize += continuousVarEncodingNumBits
			else:
				self.encodedFeatureSize = config.numberOfFeatures * continuousVarEncodingNumBits
		prevSize = self.encodedFeatureSize
		print("self.encodedFeatureSize = ", self.encodedFeatureSize)

		# -----------------------------
		# Hidden connection matrices
		# -----------------------------
		self.hiddenConnectionMatrix: List[torch.Tensor] = []
		self.hiddenConnectionMatrixExcitatory: List[torch.Tensor] = []
		self.hiddenConnectionMatrixInhibitory: List[torch.Tensor] = []

		for layerIdx in range(config.numberOfHiddenLayers):
			if self.useEIneurons: # Corrected: use self.useEIneurons
				excitSize = config.hiddenLayerSize // 2
				inhibSize = config.hiddenLayerSize - excitSize
				excMat = self._initialise_layer_weights(excitSize, prevSize, layerIdx) # Corrected: added layerIdx
				inhMat = self._initialise_layer_weights(inhibSize, prevSize, layerIdx) # Corrected: added layerIdx
				self.hiddenConnectionMatrixExcitatory.append(excMat)
				self.hiddenConnectionMatrixInhibitory.append(inhMat)
			else:
				mat = self._initialise_layer_weights(config.hiddenLayerSize, prevSize, layerIdx) # Corrected: added layerIdx
				self.hiddenConnectionMatrix.append(mat)
			prevSize = config.hiddenLayerSize

		if self.useDynamicGeneratedHiddenConnections:
			# self.register_buffer('neuronSegmentAssignedMask', torch.zeros(config.numberOfHiddenLayers, config.hiddenLayerSize, dtype=torch.bool, device=device)) # Reverted
			self.neuronSegmentAssignedMask = torch.zeros(config.numberOfHiddenLayers, config.hiddenLayerSize, dtype=torch.bool, device=device) # Ensure device

		# -----------------------------
		# Output connection matrix
		# -----------------------------
		if self.useOutputConnectionsLastLayer:
			outConnShape = (config.hiddenLayerSize, config.numberOfClasses,)
		else:
			outConnShape = (config.numberOfHiddenLayers, config.hiddenLayerSize, config.numberOfClasses,)
		if useBinaryOutputConnections:
			self.outputConnectionMatrix = torch.zeros(outConnShape, dtype=torch.bool, device=device) # Added device=device
		else:
			self.outputConnectionMatrix = torch.zeros(outConnShape, dtype=torch.float, device=device) # Added device=device

		
		# -----------------------------
		# verify neuron uniqueness
		# -----------------------------
		if(useDynamicGeneratedHiddenConnections and useDynamicGeneratedHiddenConnectionsUniquenessChecks):
			L = self.config.numberOfHiddenLayers
			self.hiddenHashes       = [torch.empty(0, dtype=torch.int64, device=device) for _ in range(L)]
			if self.useEIneurons:
				self.hiddenHashesExc = [torch.empty(0, dtype=torch.int64, device=device) for _ in range(L)]
				self.hiddenHashesInh = [torch.empty(0, dtype=torch.int64, device=device) for _ in range(L)]
			'''
			self.hiddenNeuronSignatures = [dict() for _ in range(config.numberOfHiddenLayers+1)]
			if self.useEIneurons:
				self.hiddenNeuronSignaturesExc = [dict() for _ in range(config.numberOfHiddenLayers+1)]
				self.hiddenNeuronSignaturesInh = [dict() for _ in range(config.numberOfHiddenLayers+1)]
			'''

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
		  If sparse and bool: True for +1, False for -1 (non-EI) or True for +1 (EI)
		  If dense and int8: +1, -1, or 0.
		"""
		cfg = self.config
		dev = device  # ensures GPU/CPU consistency
		k = cfg.numberOfSynapsesPerSegment  # shorthand

		if self.useSparseMatrix:
			# -------------------------------------------------- sparse initialisation
			sparse_dtype = torch.bool
			if self.useDynamicGeneratedHiddenConnections:
				# start EMPTY
				indices = torch.empty((2, 0), dtype=torch.long, device=dev)
				values  = torch.empty((0,), dtype=sparse_dtype, device=dev)
			else:
				# start with k random synapses per neuron
				row_idx = torch.arange(numNeurons, device=dev).repeat_interleave(k)
				col_idx = torch.randint(prevSize, (numNeurons * k,), device=dev)
				indices = torch.stack([row_idx, col_idx])  # shape [2, nnz]

				if self.useEIneurons:
					values = torch.ones(indices.size(1), device=dev, dtype=sparse_dtype) # All True
				else:
					# True/False with equal probability
					values = torch.randint(0, 2, (indices.size(1),), device=dev, dtype=sparse_dtype)

			mat = torch.sparse_coo_tensor(indices, values, size=(numNeurons, prevSize), device=dev, dtype=sparse_dtype,).coalesce()
			return mat
		else:
			# -------------------------------------------------- dense initialisation
			weight = torch.zeros(numNeurons, prevSize, device=dev, dtype=torch.int8) # Use torch.int8 for dense

			if not self.useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell
				for n in range(numNeurons):
					syn_idx = torch.randperm(prevSize, device=dev)[:k]

					if self.useEIneurons:
						weight[n, syn_idx] = 1
					else:
						# Generate +1 or -1 with equal probability, as int8
						rand_signs_bool = torch.randint(0, 2, (k,), device=dev, dtype=torch.bool)
						rand_signs = torch.where(rand_signs_bool, 
						                        torch.tensor(1, device=dev, dtype=torch.int8), 
						                        torch.tensor(-1, device=dev, dtype=torch.int8))
						weight[n, syn_idx] = rand_signs

			# make it a learnable parameter for the dense case
			return nn.Parameter(weight)

	# ---------------------------------------------------------
	# Forward pass
	# ---------------------------------------------------------

	@torch.no_grad()
	def forward(self, trainOrTest: bool, x: torch.Tensor, y: Optional[torch.Tensor] = None, optim=None, l=None, fieldTypeList=None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass.
		
		Args:
			trainOrTest: True=> train mode; False=> inference.
			x: (batch, features) tensor in continuous range.
			y: (batch,) int64 labels when trainOrTest==True.

		Returns:
			predictions, outputActivations (both shape (batch, classes)).
		"""
		batchSize = x.size(0)
		#assert (batchSize == self.config.batchSize), "Batch size must match config.batchSize"
		device = x.device

		encoded = self._encodeContinuousVarsAsBits(x)

		if useImageDataset:
			# -----------------------------
			# Convolve inputs
			# -----------------------------
			prevActivation = EISANIpt_EISANImodelCNN._propagate_conv_layers(self, encoded)	# (batch, encodedFeatureSize) int8
		else:
			prevActivation = encoded.to(torch.int8)
		
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
						EISANIpt_EISANImodelDynamic._dynamic_hidden_growth_vectorised(self, layerIdx, prevActivation, currentActivation, device)
					else:
						for s in range(prevActivation.size(0)):                # loop over batch
							prevAct_b  = prevActivation[s : s + 1]             # keep 2- [1, prevSize]
							currAct_b  = currentActivation[s : s + 1]          # keep 2- [1, layerSize]
							EISANIpt_EISANImodelDynamic._dynamic_hidden_growth(self, layerIdx, prevAct_b, currAct_b, device)

			prevActivation = currentActivation

		# -----------------------------
		# Output layer
		# -----------------------------
		outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

		if self.useOutputConnectionsLastLayer:
			lastLayerActivation = layerActivations[-1] # Activations from the last hidden layer
			if self.useBinaryOutputConnections:
				weights = self.outputConnectionMatrix.to(torch.int8) # bool to int8 (0/1)
				outputActivations += lastLayerActivation.float() @ weights.float()
			else:
				weights = self.outputConnectionMatrix # float
				outputActivations += lastLayerActivation.float() @ weights
			
			if trainOrTest and y is not None:
				# For the last layer, layerIdx is effectively self.config.numberOfHiddenLayers - 1
				# but since outputConnectionMatrix is 2D, we don't pass layerIdx or pass a dummy one if the function expects it
				self._update_output_connections(self.config.numberOfHiddenLayers - 1, lastLayerActivation, y, device)
		else:
			for layerIdx, act in enumerate(layerActivations): # act is torch.int8
				if self.useBinaryOutputConnections:
					weights = self.outputConnectionMatrix[layerIdx].to(torch.int8) # bool to int8 (0/1)
					outputActivations += act.float() @ weights.float() # Cast to float for matmul
				else:
					weights = self.outputConnectionMatrix[layerIdx] # float
					outputActivations += act.float() @ weights # cast act to float for matmul

				# Training: reinforce output connections
				if trainOrTest and y is not None:
					self._update_output_connections(layerIdx, act, y, device)

		predictions = torch.argmax(outputActivations, dim=1)
		
		# count how many are exactly correct
		correct = (predictions == y).sum().item()
		accuracy = correct / y.size(0)
		loss = Loss(0.0)
		
		return loss, accuracy

	# -------------------------------------------------------------
	# Encoding helpers
	# -------------------------------------------------------------

	def _encodeContinuousVarsAsBits(self, x: torch.Tensor) -> torch.Tensor:
		if(useImageDataset):
			numBits = EISANICNNcontinuousVarEncodingNumBits
			if numBits == 1:
				return x
			B, C, H, W = x.shape
			x = x.view(B, C * H * W)  # Flatten pixel dimensions
		else:
			numBits = continuousVarEncodingNumBits

		if useGrayCode:
			encoded_bits_list = self._gray_code_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		else:
			encoded_bits_list = self._thermometer_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)

		if(useImageDataset):
			code = torch.stack(encoded_bits_list, dim=2) 	#[B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
			code = code.view(B, C, H, W, numBits)	#unflatten pixel dimensions
			code = code.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, C, EISANICNNcontinuousVarEncodingNumBits, H, W]
			code = code.reshape(B, C*numBits, H, W)
		else:
			code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension	#[B, nCont*EISANICNNcontinuousVarEncodingNumBits]

		return code

	def _gray_code_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
		"""Vectorised Gray-code encoding (batch, features, bits) with no costly Python loop."""
		batch_size, num_features = x.shape
		device = x.device

		# 1. Identify Boolean columns once
		isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
		if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
			limit = min(num_features, len(fieldTypeList))
			isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

		# 2. Boolean bits (1-bit, already 0/1)
		boolBits = x[:, isBool].float()											# (batch, nBool)

		# 3. Continuous bits - quantise - Gray - bit-slice
		if (~isBool).any():
			xCont = x[:, ~isBool]												# (batch, nCont)
			intLevels = self._continuous_to_int(xCont, numBits, minVal, maxVal)		# (batch, nCont)
			grayLevels = intLevels ^ (intLevels >> 1)
			bitPos = torch.arange(numBits, device=device)						# (numBits,)
			contBits = ((grayLevels.unsqueeze(-1) >> bitPos) & 1).float()		# (batch, nCont, numBits)
		else:
			contBits = x.new_empty(batch_size, 0, numBits)

		# 4. Re-interleave (cheap Python, negligible cost)
		encoded_bits_list: list[torch.Tensor] = []
		bool_iter = iter(boolBits.unbind(dim=1))									# each - (batch,)
		cont_iter = iter(contBits.unbind(dim=1))								# each - (batch, numBits)
		for flag in isBool:														# trivial loop
			encoded_bits_list.append(next(bool_iter) if flag else next(cont_iter))
		return encoded_bits_list

	def _thermometer_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
		"""Vectorised thermometer encoding (batch, features, bits) without per-feature Python work."""
		batch_size, num_features = x.shape
		device = x.device

		isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
		if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
			limit = min(num_features, len(fieldTypeList))
			isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

		boolBits = x[:, isBool].float()											# (batch, nBool)

		if (~isBool).any():
			xCont = x[:, ~isBool]												# (batch, nCont)
			intLevels = self._continuous_to_int(xCont, numBits, minVal, maxVal)		# (batch, nCont)
			thresholds = torch.arange(numBits, device=device)					# (numBits,)
			contBits = (intLevels.unsqueeze(-1) >= thresholds).float()			# (batch, nCont, numBits)
		else:
			contBits = x.new_empty(batch_size, 0, numBits)

		encoded_bits_list: list[torch.Tensor] = []
		bool_iter = iter(boolBits.unbind(dim=1))
		cont_iter = iter(contBits.unbind(dim=1))
		for flag in isBool:
			encoded_bits_list.append(next(bool_iter) if flag else next(cont_iter))
		return encoded_bits_list

	def _continuous_to_int(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
		"""Map (batch, features) continuous tensor to integer levels [0, 2**bits-1]."""
		xClamped = torch.clamp(x, minVal, maxVal)
		norm = (xClamped - minVal) / (maxVal - minVal)
		scaled = norm * (2 ** numBits - 1)
		return scaled.round().to(torch.long)


	# ---------------------------------------------------------
	# Layer computation helpers
	# ---------------------------------------------------------

	def _compute_layer_standard(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:
		# prevActivation is torch.int8 (0 or 1)
		weight = self.hiddenConnectionMatrix[layerIdx].to(device)
		
		dev	= prevActivation.device
		# weight = self.hiddenConnectionMatrix[layerIdx].to(dev) # Already done above
		
		if weight.is_sparse:
			# Called only when self.useEIneurons is False.
			# Sparse bool weights: True is +1, False is -1.
			indices = weight.indices()
			values = weight.values() # bool
			numeric_values_float = torch.where(values, torch.tensor(1.0, device=dev, dtype=torch.float32), torch.tensor(-1.0, device=dev, dtype=torch.float32))
			weight_eff_float = torch.sparse_coo_tensor(indices, numeric_values_float, weight.shape, device=dev, dtype=torch.float32).coalesce()
			z_float = torch.sparse.mm(weight_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense weights are int8: +1, -1, or 0.
			z_float = prevActivation.float() @ weight.float().t() # Cast both to float for matmul
		
		activated = (z_float >= self.segmentActivationThreshold).to(torch.int8) # bool to int8 (0 or 1)
		return activated

	def _compute_layer_EI(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
		# prevActivation is torch.int8 (0 or 1)
		dev  = prevActivation.device
		wExc = self.hiddenConnectionMatrixExcitatory[layerIdx].to(dev)
		wInh = self.hiddenConnectionMatrixInhibitory[layerIdx].to(dev)
		
		# Excitatory
		if wExc.is_sparse:
			# EI sparse weights are True for +1
			numeric_values_exc_float = wExc.values().to(torch.float32) # True becomes 1.0
			wExc_eff_float = torch.sparse_coo_tensor(wExc.indices(), numeric_values_exc_float, wExc.shape, device=dev, dtype=torch.float32).coalesce()
			zExc_float = torch.sparse.mm(wExc_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense EI weights are 1 (int8). Convert to float for matmul.
			zExc_float = prevActivation.float() @ wExc.float().t()
		aExc = (zExc_float >= self.segmentActivationThreshold).to(torch.int8) # bool to int8 (0 or 1)
		
		# Inhibitory
		if wInh.is_sparse:
			# EI sparse weights are True for +1
			numeric_values_inh_float = wInh.values().to(torch.float32) # True becomes 1.0
			wInh_eff_float = torch.sparse_coo_tensor(wInh.indices(), numeric_values_inh_float, wInh.shape, device=dev, dtype=torch.float32).coalesce()
			zInh_float = torch.sparse.mm(wInh_eff_float, prevActivation.float().t()).t()
		else: # dense
			# Dense EI weights are 1 (int8). Convert to float for matmul.
			zInh_float = prevActivation.float() @ wInh.float().t()
		firesInh = zInh_float >= self.segmentActivationThreshold
		aInh = torch.zeros_like(zInh_float, dtype=torch.int8, device=dev) # Initialize with correct shape, device and int8 type
		aInh[firesInh] = -1
		return aExc, aInh


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
			if self.useOutputConnectionsLastLayer:
				if self.useBinaryOutputConnections:
					self.outputConnectionMatrix[activeNeurons, targetClass] = True
				else:
					self.outputConnectionMatrix[activeNeurons, targetClass] += 1.0
			else:
				if self.useBinaryOutputConnections:
					self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] = True
				else:
					self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] += 1.0

	# ---------------------------------------------------------
	# Post prune helper
	# ---------------------------------------------------------

	def executePostTrainPrune(self, trainOrTest) -> None:
		if(trainOrTest):
	
			if debugMeasureClassExclusiveNeuronRatio:
				measure_class_exclusive_neuron_ratio(self)
			if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
				measure_ratio_of_hidden_neurons_with_output_connections(self)

			if limitOutputConnectionsBasedOnPrevelanceAndExclusivity:
				prune_output_connections_based_on_prevalence_and_exclusivity(self)
				
				if debugMeasureClassExclusiveNeuronRatio:
					measure_class_exclusive_neuron_ratio(self)
				if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
					measure_ratio_of_hidden_neurons_with_output_connections(self)


def measure_ratio_of_hidden_neurons_with_output_connections(self) -> float:
	"""Compute ratio of hidden neurons having any output connection."""
	oc = self.outputConnectionMatrix
	if not self.useOutputConnectionsLastLayer:
		oc = oc.view(-1, oc.shape[-1])
	mask = oc != 0
	any_conn = mask.any(dim=1)
	if any_conn.numel() == 0:
		return 0.0
	ratio = any_conn.sum().item() / any_conn.numel()
	printf("measure_ratio_of_hidden_neurons_with_output_connections = ", ratio)
	return ratio

def measure_class_exclusive_neuron_ratio(self) -> float:
	"""Compute ratio of class-exclusive to non-class-exclusive hidden neurons."""
	oc = self.outputConnectionMatrix
	if not self.useOutputConnectionsLastLayer:
		oc = oc.view(-1, oc.shape[-1])
	mask = oc != 0
	counts = mask.sum(dim=1)
	exclusive = (counts == 1).sum().item()
	non_exclusive = (counts > 1).sum().item()
	if non_exclusive == 0:
		ratio = float('inf')
	else:
		ratio = exclusive / non_exclusive
	printf("measure_class_exclusive_neuron_ratio = ", ratio)
	return ratio

def prune_output_connections_based_on_prevalence_and_exclusivity(self) -> None:
	"""Prune output connections not both prevalent and exclusive to one class."""
	oc = self.outputConnectionMatrix
	if not self.useOutputConnectionsLastLayer:
		oc = oc.view(-1, oc.shape[-1])
	weights = oc
	prevalent = weights > limitOutputConnectionsPrevelanceMin
	exclusive = prevalent.sum(dim=1) == 1
	keep = prevalent & exclusive.unsqueeze(1)
	if useBinaryOutputConnectionsEffective:
		oc[...] = keep
	else:
		oc[...] = oc * keep.float()

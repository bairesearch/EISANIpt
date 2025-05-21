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
# Encoding helpers
# -------------------------------------------------------------

def _continuous_to_int(x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""Map (batch, features) continuous tensor to integer levels [0, 2**bits-1]."""
	xClamped = torch.clamp(x, minVal, maxVal)
	norm = (xClamped - minVal) / (maxVal - minVal)
	scaled = norm * (2 ** numBits - 1)
	return scaled.round().to(torch.long)


def gray_code_encode(x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> torch.Tensor:
	"""Vectorised Gray-code encoding (batch, features, bits)."""
	batch_size, num_features = x.shape
	encoded_bits_list = []
	singleBits = 0
	for i in range(num_features):
		if encodeDatasetBoolValuesAs1Bit and i < len(fieldTypeList) and fieldTypeList[i] == 'bool':
			# Boolean field, encode as 1 bit
			bit = x[:, i:i+1].float()	# Ensure float, shape (batch, 1)
			#print("bit.shape = ", bit.shape)
			encoded_bits_list.append(bit) 
			singleBits = singleBits + 1
		else:
			# Continuous field, use Gray code
			intLevels = _continuous_to_int(x[:, i:i+1], numBits, minVal, maxVal)
			grayLevels = intLevels ^ (intLevels >> 1)
			bitPositions = torch.arange(numBits, device=x.device)
			bits = ((grayLevels >> bitPositions) & 1).float()
			#print("bits.shape = ", bits.shape)
			encoded_bits_list.append(bits)

	code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension
	#print("code.shape = ", code.shape)
	#print("singleBits = ", singleBits)
	return code

def thermometer_encode(x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> torch.Tensor:
	"""Vectorised thermometer encoding (batch, features, bits)."""
	batch_size, num_features = x.shape
	encoded_bits_list = []

	for i in range(num_features):
		if encodeDatasetBoolValuesAs1Bit and i < len(fieldTypeList) and fieldTypeList[i] == 'bool':
			# Boolean field, encode as 1 bit
			bit = x[:, i:i+1].float()	# Ensure float, shape (batch, 1)
			encoded_bits_list.append(bit) 
		else:
			# Continuous field, use thermometer encoding
			intLevels = _continuous_to_int(x[:, i:i+1], numBits, minVal, maxVal)
			thresholds = torch.arange(numBits, device=x.device)
			bits = (intLevels >= thresholds).float()
			encoded_bits_list.append(bits)

	code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension
	return code

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
		self.useOutputConnectionsLastLayer = useOutputConnectionsLastLayer

		# -----------------------------
		# Derived sizes
		# -----------------------------
		if useImageDataset:
			EISANIpt_EISANImodelCNN._init_conv_layers(self)                  # fills self.convKernels & self.encodedFeatureSize
			prevSize = self.encodedFeatureSize
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
			self.hiddenNeuronSignatures = [dict() for _ in range(config.numberOfHiddenLayers+1)]
			if self.useEIneurons:
				self.hiddenNeuronSignaturesExc = [dict() for _ in range(config.numberOfHiddenLayers+1)]
				self.hiddenNeuronSignaturesInh = [dict() for _ in range(config.numberOfHiddenLayers+1)]


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
		assert (batchSize == self.config.batchSize), "Batch size must match config.batchSize"
		device = x.device

		if useImageDataset:
			# -----------------------------
			# Convolve inputs
			# -----------------------------
			prevActivation = EISANIpt_EISANImodelCNN._propagate_conv_layers(self, x)	# (batch, encodedFeatureSize) int8
		else:
			# -----------------------------
			# Encode inputs
			# -----------------------------
			if self.useGrayCode:
				encoded = gray_code_encode(x, self.continuousVarEncodingNumBits, self.continuousVarMin, self.continuousVarMax, fieldTypeList)
			else:
				encoded = thermometer_encode(x, self.continuousVarEncodingNumBits, self.continuousVarMin, self.continuousVarMax, fieldTypeList)
			# Flatten feature & bit dimensions -> (batch, encodedFeatureSize)
			prevActivation = encoded.view(batchSize, -1).to(torch.int8)
		
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
				EISANIpt_EISANImodelDynamic.measure_class_exclusive_neuron_ratio(self)
			if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
				EISANIpt_EISANImodelDynamic.measure_ratio_of_hidden_neurons_with_output_connections(self)

			if limitOutputConnectionsBasedOnPrevelanceAndExclusivity:
				EISANIpt_EISANImodelDynamic.prune_output_connections_based_on_prevalence_and_exclusivity(self)
				
				if debugMeasureClassExclusiveNeuronRatio:
					EISANIpt_EISANImodelDynamic.measure_class_exclusive_neuron_ratio(self)
				if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
					EISANIpt_EISANImodelDynamic.measure_ratio_of_hidden_neurons_with_output_connections(self)


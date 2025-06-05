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
import EISANIpt_EISANI_globalDefs
from ANNpt_globalDefs import *
from typing import List, Optional, Tuple
if(useDynamicGeneratedHiddenConnections):
	import EISANIpt_EISANImodelDynamic
if(useImageDataset):
	import EISANIpt_EISANImodelCNN


def generateNumberHiddenLayers(numberOfLayers: int, numberOfConvlayers: int) -> None:
	if(useImageDataset):
		numberOfLinearLayers = numberOfLayers - numberOfConvlayers
		numberOfHiddenLayers = numberOfLinearLayers - 1
	else:
		numberOfHiddenLayers = numberOfLayers - 1
	return numberOfHiddenLayers

def getNumberUniqueLayers(recursiveLayers, recursiveSuperblocksNumber, numberOfHiddenLayers):
	if(recursiveLayers):
		numberUniqueLayers = recursiveSuperblocksNumber*2
		#*2 explanation: the first forward propagated layer in a superblock always uses unique weights:
		#	- for the first superblock this will comprise unique weights between the input layer and the first hidden layer of the superblock
		#	- for the all other superblocks this will comprise unique weights between the previous superblock and the first hidden layer of the superblock
	else:
		numberUniqueLayers = numberOfHiddenLayers
	return numberUniqueLayers

def generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers):
	numberOfHiddenLayers = generateNumberHiddenLayers(numberOfLayers, numberOfConvlayers)
	if(useDynamicGeneratedHiddenConnections):
		datasetSizeRounded = round_up_to_power_of_2(datasetSize)
		hiddenLayerSizeSANI = hiddenLayerSizeSANIbase*datasetSizeRounded * trainNumberOfEpochs
	else:
		hiddenLayerSizeSANI = EISANIpt_EISANI_globalDefs.hiddenLayerSizeSANI
	if(recursiveLayers):
		maxNumberRecursionsAcrossHiddenLayer = numberOfHiddenLayers-1	#-1 because first forward propagated layer in a superblock always uses unique weights
		hiddenLayerSizeSANI = hiddenLayerSizeSANI * maxNumberRecursionsAcrossHiddenLayer
	return hiddenLayerSizeSANI
		
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
		self.numberOfHiddenLayers = generateNumberHiddenLayers(numberOfLayers, numberOfConvlayers) # Added
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
		self.useBinaryOutputConnectionsEffective = useBinaryOutputConnectionsEffective
		self.segmentActivationThreshold = segmentActivationThreshold
		self.targetActivationSparsityFraction = targetActivationSparsityFraction
		self.useOutputConnectionsLastLayer = useOutputConnectionsLastLayer
		self.numberOfSegmentsPerNeuron = numberOfSegmentsPerNeuron # Added
		self.recursiveLayers = recursiveLayers # Added
		self.recursiveSuperblocksNumber = recursiveSuperblocksNumber # Added

		# -----------------------------
		# Derived sizes
		# -----------------------------

		self.numberUniqueLayers = getNumberUniqueLayers(self.recursiveLayers, self.recursiveSuperblocksNumber, config.numberOfHiddenLayers)

		if useImageDataset:
			EISANIpt_EISANImodelCNN._init_conv_layers(self)                  # fills self.convKernels & self.encodedFeatureSize

			#if(EISANICNNdynamicallyGenerateLinearInputFeatures):	
			#	self.CNNoutputEncodedFeaturesDict = {} #key: CNNoutputLayerFlatFeatureIndex, value: linearInputLayerFeatureIndex	#non-vectorised implementation
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

		for layerIdx in range(self.numberUniqueLayers): # Modified
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
			self.neuronSegmentAssignedMask = torch.zeros(self.numberUniqueLayers, config.hiddenLayerSize, self.numberOfSegmentsPerNeuron, dtype=torch.bool, device=device) # Ensure device, Added numberOfSegmentsPerNeuron, Modified

		# -----------------------------
		# Output connection matrix
		# -----------------------------
		if self.useOutputConnectionsLastLayer:
			outConnShape = (config.hiddenLayerSize, config.numberOfClasses,)
		else:
			outConnShape = (self.numberUniqueLayers, config.hiddenLayerSize, config.numberOfClasses,) # Modified
		if useBinaryOutputConnections:
			self.outputConnectionMatrix = torch.zeros(outConnShape, dtype=torch.bool, device=device) # Added device=device
		else:
			self.outputConnectionMatrix = torch.zeros(outConnShape, dtype=torch.float, device=device) # Added device=device


		# -------------------------------------------------------------
		# Hidden-neuron prediction-accuracy tracker
		#   dim-2: 0 -> #correct, 1 -> #total
		# -------------------------------------------------------------
		if(limitOutputConnectionsBasedOnAccuracy):
			self.hiddenNeuronPredictionAccuracy = torch.zeros(self.numberUniqueLayers, config.hiddenLayerSize, 2, dtype=torch.int64, device=device,)	#2: (correct, total)

		# -----------------------------
		# verify neuron uniqueness
		# -----------------------------
		if(useDynamicGeneratedHiddenConnections and useDynamicGeneratedHiddenConnectionsUniquenessChecks):
			L = self.numberUniqueLayers # Modified
			S = self.numberOfSegmentsPerNeuron # Added
			'''
			self.hiddenHashes       = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
			if self.useEIneurons:
				self.hiddenHashesExc = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
				self.hiddenHashesInh = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
			'''
			self.hiddenNeuronSignatures = [[dict() for _ in range(S)] for _ in range(L+1)]
			if self.useEIneurons:
				self.hiddenNeuronSignaturesExc = [[dict() for _ in range(S)] for _ in range(L+1)]
				self.hiddenNeuronSignaturesInh = [[dict() for _ in range(S)] for _ in range(L+1)]
		

	# ---------------------------------------------------------
	# Helper - layer initialisation
	# ---------------------------------------------------------

	def getUniqueLayerIndex(self, layerIdSuperblock: int, layerIdHidden: int) -> int: # Added
		if(self.recursiveLayers):
			if(layerIdHidden==0):
				uniqueLayerIndex = layerIdSuperblock*2
			else:
				uniqueLayerIndex = layerIdSuperblock*2+1
		else:
			uniqueLayerIndex = layerIdHidden
		return uniqueLayerIndex

	def _initialise_layer_weights(self, numNeurons: int, prevSize: int, layerIdx: int,) -> torch.Tensor:
		"""
		Create the hidden connection matrix for a single layer.

		- If `self.useDynamicGeneratedHiddenConnections` is True  -> start empty
		- Else													-> randomly connect
		  exactly `self.cfg.numberOfSynapsesPerSegment` synapses
		  per neuron segment. # Modified

		Weight values:
		  - EI-mode  ->  +1  (excitatory) or always-on +1 (inhibitory handled elsewhere)
		  - Standard ->  1 with equal probability
		  If sparse and bool: True for +1, False for -1 (non-EI) or True for +1 (EI)
		  If dense and int8: +1, -1, or 0.
		Shape: [numNeurons, numberOfSegmentsPerNeuron, prevSize] # Added
		"""
		cfg = self.config
		dev = device  # ensures GPU/CPU consistency
		k = cfg.numberOfSynapsesPerSegment  # shorthand
		s = self.numberOfSegmentsPerNeuron # shorthand, Added
		nnz_per_segment = k # numNeurons * k # Modified: nnz is per segment now for sparse
		nnz = numNeurons * s * k # Total non-zero elements if all segments initialized

		if self.useSparseMatrix:

			sparse_dtype = torch.bool
			if(initialiseSANIlayerWeightsUsingCPU):
				devTemp =  pt.device('cpu')
			else:
				devTemp = dev
				
			if self.useDynamicGeneratedHiddenConnections:
				# start EMPTY
				indices = torch.empty((3, 0), dtype=torch.int64, device=devTemp) # Modified for 3D sparse
				values  = torch.empty((0,),  dtype=sparse_dtype, device=devTemp)
				mat_shape = (numNeurons, s, prevSize) # Added
			else:
				# start with k random synapses per neuron segment
				# indices will be [neuron_idx, segment_idx, prev_neuron_idx]
				neuron_indices = torch.arange(numNeurons, device=devTemp).repeat_interleave(s * k)
				segment_indices = torch.arange(s, device=devTemp).repeat_interleave(k).repeat(numNeurons)
				col_indices = torch.randint(prevSize, (nnz,), device=devTemp, dtype=torch.int64)
				
				indices = torch.stack([neuron_indices, segment_indices, col_indices], dim=0) # shape [3, nnz]
				mat_shape = (numNeurons, s, prevSize) # Added

				if self.useEIneurons:
					values = torch.ones(nnz, device=devTemp, dtype=sparse_dtype)
				else:
					values = torch.randint(0, 2, (nnz,), device=devTemp, dtype=sparse_dtype)
         
			mat = torch.sparse_coo_tensor(indices, values, size=mat_shape, device=devTemp, dtype=sparse_dtype,).coalesce() # Modified
			mat = mat.to(dev)
			return mat
		else:
			# -------------------------------------------------- dense initialisation
			weight_shape = (numNeurons, s, prevSize) # Modified
			weight = torch.zeros(weight_shape, device=dev, dtype=torch.int8) # Use torch.int8 for dense

			if not self.useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell segment
				for n in range(numNeurons):
					for seg_idx in range(s): # Added loop for segments
						syn_idx = torch.randperm(prevSize, device=dev)[:k]

						if self.useEIneurons:
							weight[n, seg_idx, syn_idx] = 1 # Modified
						else:
							# Generate +1 or -1 with equal probability, as int8
							rand_signs_bool = torch.randint(0, 2, (k,), device=dev, dtype=torch.bool)
							rand_signs = torch.where(rand_signs_bool, torch.tensor(1, device=dev, dtype=torch.int8), torch.tensor(-1, device=dev, dtype=torch.int8))
							weight[n, seg_idx, syn_idx] = rand_signs # Modified

			return weight

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

		for layerIdSuperblock in range(self.recursiveSuperblocksNumber): # Modified
			for layerIdHidden in range(self.config.numberOfHiddenLayers): # Modified
				uniqueLayerIndex = self.getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
				if self.useEIneurons:
					aExc, aInh = self._compute_layer_EI(uniqueLayerIndex, prevActivation, device) # Modified
					currentActivation = torch.cat([aExc, aInh], dim=1)
				else:
					currentActivation = self._compute_layer_standard(uniqueLayerIndex, prevActivation, device) # Modified

				layerActivations.append(currentActivation)

				# -------------------------
				# Dynamic hidden connection growth
				# -------------------------
				if (trainOrTest and self.useDynamicGeneratedHiddenConnections):
					for _ in range(numberNeuronsGeneratedPerSample):
						if(useDynamicGeneratedHiddenConnectionsVectorised):
							EISANIpt_EISANImodelDynamic._dynamic_hidden_growth_vectorised(self, uniqueLayerIndex, prevActivation, currentActivation, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified
						else:
							for s_batch_idx in range(prevActivation.size(0)):                # loop over batch
								prevAct_b  = prevActivation[s_batch_idx : s_batch_idx + 1]             # keep 2- [1, prevSize]
								currAct_b  = currentActivation[s_batch_idx : s_batch_idx + 1]          # keep 2- [1, layerSize]
								EISANIpt_EISANImodelDynamic._dynamic_hidden_growth(self, uniqueLayerIndex, prevAct_b, currAct_b, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified

				prevActivation = currentActivation

		# -----------------------------
		# Output layer
		# -----------------------------
		outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

		if self.useOutputConnectionsLastLayer:
			lastLayerActivation = layerActivations[-1] # Activations from the last hidden layer
			
			if self.useBinaryOutputConnectionsEffective:
				weights = self.outputConnectionMatrix.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
			else:
				if self.useBinaryOutputConnections:
					weights = self.outputConnectionMatrix.to(torch.int8) # bool to int8 (0/1)
				else:
					weights = self.outputConnectionMatrix # float
			outputActivations += lastLayerActivation.float() @ weights.float()
			
			if trainOrTest and y is not None:
				# For the last layer, layerIdx is effectively self.config.numberOfHiddenLayers - 1
				# but since outputConnectionMatrix is 2D, we don't pass layerIdx or pass a dummy one if the function expects it
				#TODO: this is probably wrong for recursive layers; should be uniqueLayerIndex?
				# Determine the correct uniqueLayerIndex for the last layer in the last superblock
				lastSuperblockIndex = self.recursiveSuperblocksNumber - 1
				lastHiddenLayerInSuperblockIndex = self.config.numberOfHiddenLayers -1
				uniqueLayerIndexForLastLayer = self.getUniqueLayerIndex(lastSuperblockIndex, lastHiddenLayerInSuperblockIndex)
				self._update_output_connections(uniqueLayerIndexForLastLayer, lastLayerActivation, y, device)
		else:
			actLayerIndex = 0 # Added
			for layerIdSuperblock in range(self.recursiveSuperblocksNumber): # Added
				for layerIdHidden in range(self.config.numberOfHiddenLayers): # Added
					act = layerActivations[actLayerIndex] # Modified
					uniqueLayerIndex = self.getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
					weights = self.outputConnectionMatrix[uniqueLayerIndex] # Modified
					
					if self.useBinaryOutputConnectionsEffective:
						weights = weights.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
					else:
						if self.useBinaryOutputConnections:
							weights = weights.to(torch.int8) # bool to int8 (0/1)
						else:
							weights = weights
					outputActivations += act.float() @ weights.float() # cast act to float for matmul

					# Training: reinforce output connections
					if trainOrTest and y is not None:
						self._update_output_connections(uniqueLayerIndex, act, y, device) # Modified
					actLayerIndex += 1 # Added

		predictions = torch.argmax(outputActivations, dim=1)

		# -----------------------------------------------------------------
		# Update hidden-neuron accuracy statistics (per-neuron definition)
		# -----------------------------------------------------------------
		if(limitOutputConnectionsBasedOnAccuracy and trainOrTest):
			if y is not None:												# labels available
				if self.useOutputConnectionsLastLayer:
					# each neuron appears only once (final hidden layer)
					neuron_to_class = torch.argmax(self.outputConnectionMatrix, dim=1)		# [H]
					has_conn        = (self.outputConnectionMatrix != 0).any(dim=1)			# [H]
				else:
					# per-layer output matrices
					neuron_to_class = torch.argmax(self.outputConnectionMatrix, dim=2)		# [L,H]
					has_conn        = (self.outputConnectionMatrix != 0).any(dim=2)			# [L,H]

				for lidx, act in enumerate(layerActivations):
					active_mask = (act != 0)												# [B,H]

					if self.useOutputConnectionsLastLayer:
						pred_class_layer = neuron_to_class
						conn_layer       = has_conn
					else:
						pred_class_layer = neuron_to_class[lidx]								# [H]
						conn_layer       = has_conn[lidx]									# [H]

					# Only neurons that have at least one output connection participate
					valid_neuron_mask = conn_layer.unsqueeze(0)								# [1,H] \u2192 broadcast

					label_eq_pred = (pred_class_layer.unsqueeze(0) == y.unsqueeze(1))		# [B,H]

					correct_neuron = active_mask & valid_neuron_mask & label_eq_pred			# [B,H]
					considered     = active_mask & valid_neuron_mask							# [B,H]

					self.hiddenNeuronPredictionAccuracy[lidx,:,0] += correct_neuron.sum(dim=0)
					self.hiddenNeuronPredictionAccuracy[lidx,:,1] += considered.sum(dim=0)

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
		# prevActivation is torch.int8 (0 or 1), shape [B, prevSize]
		weight_matrix = self.hiddenConnectionMatrix[layerIdx].to(device) # shape [numNeurons, numSegments, prevSize]
		
		dev	= prevActivation.device
		numNeurons, numSegments, prevSize = weight_matrix.shape
		B = prevActivation.shape[0]

		# Reshape prevActivation to be [B, 1, 1, prevSize] for broadcasting with weights [1, numNeurons, numSegments, prevSize]
		# Or, more efficiently, iterate or use batch matrix multiply if applicable after reshaping.
		# For sparse: iterate over segments or adapt sparse.mm if possible for 3D.
		# For dense: use bmm or einsum.
		
		# prevActivation: [B, prevSize]
		# weight_matrix: [numNeurons, numSegments, prevSize] (dense) or sparse equivalent
		
		if weight_matrix.is_sparse:
			# Sparse multiplication per segment and then combine.
			# This is a simplified approach. Efficient sparse 3D multiplication is complex.
			segment_activations_list = []
			weight_matrix_coalesced = weight_matrix.coalesce()
			for s_idx in range(numSegments):
				# Extract segment specific weights. This requires careful handling of sparse indices.
				# This is a placeholder for a more efficient sparse slicing/multiplication.
				# Assuming a method to get a 2D sparse matrix for each segment: [numNeurons, prevSize]
				# For simplicity, let's imagine we can filter indices for the current segment.
				# This part needs a robust way to handle 3D sparse tensors per segment.
				
				# Simplified: Reconstruct a 2D sparse matrix for each segment
				# This is inefficient and for illustration only.
				indices_3d = weight_matrix_coalesced.indices() # [3, nnz]
				values_3d = weight_matrix_coalesced.values()   # [nnz]
				
				mask_segment = indices_3d[1] == s_idx
				if mask_segment.any():
					indices_2d_segment = indices_3d[[0, 2]][:, mask_segment] # neuron_idx, prev_neuron_idx
					values_2d_segment = values_3d[mask_segment]
					
					# Ensure indices are within bounds for a [numNeurons, prevSize] matrix
					indices_2d_segment[0] = torch.clamp(indices_2d_segment[0], 0, numNeurons -1)
					indices_2d_segment[1] = torch.clamp(indices_2d_segment[1], 0, prevSize -1)

					weight_segment_sparse = torch.sparse_coo_tensor(indices_2d_segment, values_2d_segment, size=(numNeurons, prevSize), device=dev, dtype=weight_matrix.dtype).coalesce()

					numeric_values_float = torch.where(weight_segment_sparse.values(), torch.tensor(1.0, device=dev, dtype=torch.float32), torch.tensor(-1.0, device=dev, dtype=torch.float32))
					weight_eff_float = torch.sparse_coo_tensor(weight_segment_sparse.indices(), numeric_values_float, weight_segment_sparse.shape, device=dev, dtype=torch.float32).coalesce()
					z_segment_float = torch.sparse.mm(weight_eff_float, prevActivation.float().t()).t() # [B, numNeurons]
				else:
					z_segment_float = torch.zeros(B, numNeurons, device=dev, dtype=torch.float32)
				segment_activations_list.append(z_segment_float.unsqueeze(2)) # [B, numNeurons, 1]
			
			z_float_all_segments = torch.cat(segment_activations_list, dim=2) # [B, numNeurons, numSegments]
		else: # dense
			# weight_matrix: [numNeurons, numSegments, prevSize]
			# prevActivation: [B, prevSize]
			# We want z_float: [B, numNeurons, numSegments]
			# z_float[b, n, s] = sum_p ( prevActivation[b, p] * weight_matrix[n, s, p] )
			z_float_all_segments = torch.einsum('bp,nsp->bns', prevActivation.float(), weight_matrix.float()) # [B, numNeurons, numSegments]
		
		# Neuron fires if ANY of its segments fire
		neuron_activated = self._neuronActivationFunction(z_float_all_segments) # [B, numNeurons] (bool)
		neuron_activated = neuron_activated.to(torch.int8) # bool to int8 (0 or 1)
		return neuron_activated

	def _compute_layer_EI(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
		# prevActivation is torch.int8 (0 or 1), shape [B, prevSize]
		dev  = prevActivation.device
		wExc_3d = self.hiddenConnectionMatrixExcitatory[layerIdx].to(dev) # [numExcNeurons, numSegments, prevSize]
		wInh_3d = self.hiddenConnectionMatrixInhibitory[layerIdx].to(dev) # [numInhNeurons, numSegments, prevSize]
		
		B = prevActivation.shape[0]
		numExcNeurons, numSegments, _ = wExc_3d.shape
		numInhNeurons, _, _ = wInh_3d.shape

		# Excitatory
		if wExc_3d.is_sparse:
			segment_activations_list_exc = []
			wExc_3d_coalesced = wExc_3d.coalesce()
			indices_3d_exc = wExc_3d_coalesced.indices()
			values_3d_exc = wExc_3d_coalesced.values()
			for s_idx in range(numSegments):
				mask_segment_exc = indices_3d_exc[1] == s_idx
				if mask_segment_exc.any():
					indices_2d_segment_exc = indices_3d_exc[[0, 2]][:, mask_segment_exc]
					values_2d_segment_exc = values_3d_exc[mask_segment_exc]
					indices_2d_segment_exc[0] = torch.clamp(indices_2d_segment_exc[0], 0, numExcNeurons -1)
					indices_2d_segment_exc[1] = torch.clamp(indices_2d_segment_exc[1], 0, prevActivation.shape[1] -1)

					wExc_segment_sparse = torch.sparse_coo_tensor(indices_2d_segment_exc, values_2d_segment_exc, size=(numExcNeurons, prevActivation.shape[1]), device=dev, dtype=wExc_3d.dtype).coalesce()
					numeric_values_exc_float = wExc_segment_sparse.values().to(torch.float32)
					wExc_eff_float = torch.sparse_coo_tensor(wExc_segment_sparse.indices(), numeric_values_exc_float, wExc_segment_sparse.shape, device=dev, dtype=torch.float32).coalesce()
					zExc_segment_float = torch.sparse.mm(wExc_eff_float, prevActivation.float().t()).t()
				else:
					zExc_segment_float = torch.zeros(B, numExcNeurons, device=dev, dtype=torch.float32)
				segment_activations_list_exc.append(zExc_segment_float.unsqueeze(2))
			zExc_float_all_segments = torch.cat(segment_activations_list_exc, dim=2) # [B, numExcNeurons, numSegments]
		else: # dense
			zExc_float_all_segments = torch.einsum('bp,nsp->bns', prevActivation.float(), wExc_3d.float()) # [B, numExcNeurons, numSegments]
		
		aExc = self._neuronActivationFunction(zExc_float_all_segments).to(torch.int8)  # [B, numExcNeurons] (0 or 1)
		
		# Inhibitory
		if wInh_3d.is_sparse:
			segment_activations_list_inh = []
			wInh_3d_coalesced = wInh_3d.coalesce()
			indices_3d_inh = wInh_3d_coalesced.indices()
			values_3d_inh = wInh_3d_coalesced.values()
			for s_idx in range(numSegments):
				mask_segment_inh = indices_3d_inh[1] == s_idx
				if mask_segment_inh.any():
					indices_2d_segment_inh = indices_3d_inh[[0, 2]][:, mask_segment_inh]
					values_2d_segment_inh = values_3d_inh[mask_segment_inh]
					indices_2d_segment_inh[0] = torch.clamp(indices_2d_segment_inh[0], 0, numInhNeurons -1)
					indices_2d_segment_inh[1] = torch.clamp(indices_2d_segment_inh[1], 0, prevActivation.shape[1] -1)

					wInh_segment_sparse = torch.sparse_coo_tensor(indices_2d_segment_inh, values_2d_segment_inh, size=(numInhNeurons, prevActivation.shape[1]), device=dev, dtype=wInh_3d.dtype).coalesce()
					numeric_values_inh_float = wInh_segment_sparse.values().to(torch.float32)
					wInh_eff_float = torch.sparse_coo_tensor(wInh_segment_sparse.indices(), numeric_values_inh_float, wInh_segment_sparse.shape, device=dev, dtype=torch.float32).coalesce()
					zInh_segment_float = torch.sparse.mm(wInh_eff_float, prevActivation.float().t()).t()
				else:
					zInh_segment_float = torch.zeros(B, numInhNeurons, device=dev, dtype=torch.float32)
				segment_activations_list_inh.append(zInh_segment_float.unsqueeze(2))
			zInh_float_all_segments = torch.cat(segment_activations_list_inh, dim=2) # [B, numInhNeurons, numSegments]
		else: # dense
			zInh_float_all_segments = torch.einsum('bp,nsp->bns', prevActivation.float(), wInh_3d.float()) # [B, numInhNeurons, numSegments]

		firesInh_neuron = self._neuronActivationFunction(zInh_float_all_segments) # [B, numInhNeurons] (bool)
		
		aInh = torch.zeros(B, numInhNeurons, dtype=torch.int8, device=dev) # Initialize with correct shape, device and int8 type
		aInh[firesInh_neuron] = -1
		return aExc, aInh

	def _segmentActivationFunction(self, z_all_segments):
		# z_all_segments has shape [B, numNeurons, numSegments]
		# A segment fires if its activation sum meets the threshold.
		segment_fires = z_all_segments >= self.segmentActivationThreshold # [B, numNeurons, numSegments] (bool)
		return segment_fires
		
	def _neuronActivationFunction(self, z_all_segments):
		segments_fires = self._segmentActivationFunction(z_all_segments)	# [B, numNeurons, numSegments] (bool)
		# Combine segment activations: a neuron is active if any of its segments are active.
		neuron_fires = torch.any(segments_fires, dim=2) # [B, numNeurons] (bool)
		return neuron_fires
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

			if limitOutputConnections:
				prune_output_connections_and_hidden_neurons(self)
				
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

################################################################################
# Output-connection pruning (iterative, prevalence / exclusivity / accuracy aware)
################################################################################
def prune_output_connections_and_hidden_neurons(self) -> None:
	"""
	Iteratively prune output connections and hidden neurons starting from the last hidden layer.

	* Last hidden layer - apply prevalence / exclusivity / accuracy tests directly.  
	* Lower layers    - apply the same tests *but* keep every neuron that still feeds any higher-layer hidden neuron.  
	* After each layer pass, call pruneHiddenNeurons() so weight matrices, signatures and masks stay consistent.

	This function assumes:
	    - when self.useOutputConnectionsLastLayer == True: self.outputConnectionMatrix is [hidden,   C]
	    - when self.useOutputConnectionsLastLayer == False: self.outputConnectionMatrix is [Lhidden, hidden, C]
	"""
	# ------------------------------------------------------------------
	def _keep_mask(lidx: int, weights: torch.Tensor) -> torch.Tensor:         # bool same shape
		
		mask = torch.ones_like(weights, dtype=torch.bool)
		if limitOutputConnectionsBasedOnPrevalence:
			prevalent = weights > limitOutputConnectionsPrevalenceMin
			mask &= prevalent

		if limitOutputConnectionsBasedOnExclusivity:
			exclusive = (weights.sum(dim=1, keepdim=True) == 1)
			mask &= exclusive
	
		if(limitOutputConnectionsBasedOnAccuracy):
			acc_layer = self.hiddenNeuronPredictionAccuracy[lidx, :, 0].float() / self.hiddenNeuronPredictionAccuracy[lidx, :, 1].clamp(min=1).float()
			accurate  = (acc_layer > limitOutputConnectionsAccuracyMin).unsqueeze(1).expand_as(weights)
			mask &= accurate
		
		return mask

	# ------------------------------------------------------------------
	def _set_kept(mat: torch.Tensor, keep: torch.Tensor) -> None:  # in-place
		if useBinaryOutputConnectionsEffective:
			mat[...] = keep
		else:
			mat.mul_(keep.to(mat.dtype))
	# ------------------------------------------------------------------
	def _still_used(layer_idx: int) -> torch.Tensor:
		"""Return bool[hidden] = neuron in <layer_idx> projects to >=1 higher layer."""
		H  = self.config.hiddenLayerSize
		dev = self.outputConnectionMatrix.device
		used = torch.zeros(H, dtype=torch.bool, device=dev)

		for upper in range(layer_idx + 1, self.numberUniqueLayers):
			if self.useEIneurons:
				mats = (self.hiddenConnectionMatrixExcitatory[upper], self.hiddenConnectionMatrixInhibitory[upper])
			else:
				mats = (self.hiddenConnectionMatrix[upper],)
			for m in mats:
				if m.is_sparse:
					prev_idx = m.coalesce().indices()[2]  # third dim indexes previous layer
					used[prev_idx.unique()] = True
				else:                                   # dense  (N, S, prevSize)
					used |= (m != 0).any(dim=0).any(dim=0)
		return used
	# ==================================================================

	if self.useOutputConnectionsLastLayer:
		# -------- Case A: only final hidden layer owns output connections ---
		oc   = self.outputConnectionMatrix                     # [hidden, C]
		keep = _keep_mask(self.numberUniqueLayers-1, oc)
		_set_kept(oc, keep)

		removed = ~(oc != 0).any(dim=1)                        # neurons now dead
		pruneHiddenNeurons(self, self.numberUniqueLayers - 1, removed)
	else:
		# -------- Case B: every hidden layer owns output connections --------
		for l in reversed(range(self.numberUniqueLayers)):
			oc_layer = self.outputConnectionMatrix[l]              # [hidden, C]
			keep = _keep_mask(l, oc_layer)

			if l < self.numberUniqueLayers - 1:                    # not topmost
				keep |= _still_used(l).unsqueeze(1)                # retain required ones

			_set_kept(oc_layer, keep)

			removed = ~(oc_layer != 0).any(dim=1)                  # [hidden]
			pruneHiddenNeurons(self, l, removed)

################################################################################
# Hidden-neuron pruning helper
################################################################################
def pruneHiddenNeurons(self, layerIndex: int, hiddenNeuronsRemoved: torch.Tensor) -> None:
	"""
	Delete (or zero-out) hidden neurons whose *output* fan-out is now zero.

	Arguments
	---------
	layerIndex : int
	    Index of the unique hidden layer being pruned.
	hiddenNeuronsRemoved : torch.BoolTensor  shape = [hidden]
	    True for every neuron that must disappear.
	"""
	if not hiddenNeuronsRemoved.any():
		return	# no work

	if(debugLimitOutputConnections):
		total_neurons  = hiddenNeuronsRemoved.numel()                   # hiddenLayerSizeSANI
		assigned_mask   = self.neuronSegmentAssignedMask[layerIndex].any(dim=1)
		assigned_count  = assigned_mask.sum().item()
		removed_assigned_mask = hiddenNeuronsRemoved & assigned_mask
		removed_count         = removed_assigned_mask.sum().item()      # \u2190 intersection
		perc_total     = removed_count / total_neurons  * 100.0
		perc_assigned  = (removed_count / assigned_count * 100.0) if assigned_count else 0.0
		printf("pruneHiddenNeurons: layer=", layerIndex, ", removed=", removed_count, "/ assigned=", assigned_count, "/ hiddenLayerSizeSANI=", total_neurons, "(", round(perc_assigned, 2), "% of assigned;", round(perc_total,   2), "% of all)")  

	dev     = hiddenNeuronsRemoved.device
	nSeg    = self.numberOfSegmentsPerNeuron
	H_total = self.config.hiddenLayerSize

	# ------------------------------------------------------------------ helpers
	def _prune_rows(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		"""Return matrix with rows in *mask* removed (sparse) or zeroed (dense)."""
		if mat.is_sparse:
			mat = mat.coalesce()
			idx = mat.indices()
			val = mat.values()
			keep = ~mask[idx[0]]
			return torch.sparse_coo_tensor(idx[:, keep], val[keep], size=mat.shape, dtype=mat.dtype, device=mat.device).coalesce()
		else:
			mat[mask] = 0
			return mat
	# ------------------------------------------------------------------
	def _purge_sigs(sig_lists, rm_mask):
		for seg in range(nSeg):
			dict_seg = sig_lists[seg]
			to_del   = []
			for sig, nid in dict_seg.items():
				idx = int(nid.item() if isinstance(nid, torch.Tensor) else nid)
				if idx < H_total and rm_mask[idx].item():
					to_del.append(sig)
			for sig in to_del:
				dict_seg.pop(sig, None)
	# ------------------------------------------------------------------

	# ---- prune hidden - hidden weight matrices ----
	if self.useEIneurons:
		H_exc = H_total // 2
		ex_rm = hiddenNeuronsRemoved[:H_exc]
		ih_rm = hiddenNeuronsRemoved[H_exc:]

		self.hiddenConnectionMatrixExcitatory[layerIndex] = _prune_rows(self.hiddenConnectionMatrixExcitatory[layerIndex], ex_rm)
		self.hiddenConnectionMatrixInhibitory[layerIndex] = _prune_rows(self.hiddenConnectionMatrixInhibitory[layerIndex], ih_rm)
	else:
		self.hiddenConnectionMatrix[layerIndex] = _prune_rows(self.hiddenConnectionMatrix[layerIndex], hiddenNeuronsRemoved)

	# ---- dynamic-growth bookkeeping -----------------------------------
	if self.useDynamicGeneratedHiddenConnections:
		self.neuronSegmentAssignedMask[layerIndex, hiddenNeuronsRemoved, :] = False

	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		if self.useEIneurons:
			H_exc = H_total // 2
			_purge_sigs(self.hiddenNeuronSignaturesExc[layerIndex], hiddenNeuronsRemoved[:H_exc])
			_purge_sigs(self.hiddenNeuronSignaturesInh[layerIndex], hiddenNeuronsRemoved[H_exc:])
		else:
			_purge_sigs(self.hiddenNeuronSignatures[layerIndex], hiddenNeuronsRemoved)

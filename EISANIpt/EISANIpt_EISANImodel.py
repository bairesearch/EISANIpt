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
if(useTabularDataset):
	pass
elif(useImageDataset):
	import EISANIpt_EISANImodelCNN
elif(useNLPDataset):
	import EISANIpt_EISANImodelNLP
import EISANIpt_EISANImodelPrune


def generateNumberHiddenLayers(numberOfLayers: int, numberOfConvlayers: int) -> None:
	if(useTabularDataset):
		numberOfHiddenLayers = numberOfLayers - 1
	elif(useImageDataset):
		numberOfLinearLayers = numberOfLayers - numberOfConvlayers
		numberOfHiddenLayers = numberOfLinearLayers - 1
	elif(useNLPDataset):
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

		# -----------------------------
		# Derived sizes
		# -----------------------------

		self.numberUniqueLayers = getNumberUniqueLayers(recursiveLayers, recursiveSuperblocksNumber, config.numberOfHiddenLayers)

		if useTabularDataset:
			# self.encodedFeatureSize = config.numberOfFeatures * EISANITABcontinuousVarEncodingNumBits # Old
			fieldTypeList = config.fieldTypeList
			if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
				self.encodedFeatureSize = 0
				for i in range(config.numberOfFeatures):
					if i < len(fieldTypeList) and fieldTypeList[i] == 'bool':
						self.encodedFeatureSize += 1
						#print("bool detected")
					else:
						self.encodedFeatureSize += EISANITABcontinuousVarEncodingNumBits
			else:
				self.encodedFeatureSize = config.numberOfFeatures * EISANITABcontinuousVarEncodingNumBits
		elif useImageDataset:
			EISANIpt_EISANImodelCNN._init_conv_layers(self)                  # fills self.convKernels & self.encodedFeatureSize

			#if(EISANICNNdynamicallyGenerateLinearInputFeatures):	
			#	self.CNNoutputEncodedFeaturesDict = {} #key: CNNoutputLayerFlatFeatureIndex, value: linearInputLayerFeatureIndex	#non-vectorised implementation
		elif useNLPDataset:
			self.encodedFeatureSize = EISANIpt_EISANImodelNLP.getEncodedFeatureSize()
		
		prevSize = self.encodedFeatureSize
		print("self.encodedFeatureSize = ", self.encodedFeatureSize)

		# -----------------------------
		# Hidden connection matrices
		# -----------------------------
		self.hiddenConnectionMatrix: List[torch.Tensor] = []
		self.hiddenConnectionMatrixExcitatory: List[torch.Tensor] = []
		self.hiddenConnectionMatrixInhibitory: List[torch.Tensor] = []

		for layerIdx in range(self.numberUniqueLayers): # Modified
			if useEIneurons: # Corrected: use useEIneurons
				if useInhibition:
					excitSize = config.hiddenLayerSize // 2
					inhibSize = config.hiddenLayerSize - excitSize
				else:
					excitSize = config.hiddenLayerSize  # All neurons are excitatory
					inhibSize = 0
				excMat = self._initialise_layer_weights(excitSize, prevSize, layerIdx) # Corrected: added layerIdx
				self.hiddenConnectionMatrixExcitatory.append(excMat)
				if useInhibition and inhibSize > 0:
					inhMat = self._initialise_layer_weights(inhibSize, prevSize, layerIdx) # Corrected: added layerIdx
					self.hiddenConnectionMatrixInhibitory.append(inhMat)
				else:
					# Create empty inhibitory matrix when inhibition is disabled
					self.hiddenConnectionMatrixInhibitory.append(torch.empty(0, numberOfSegmentsPerNeuron, prevSize, device=device))
			else:
				mat = self._initialise_layer_weights(config.hiddenLayerSize, prevSize, layerIdx) # Corrected: added layerIdx
				self.hiddenConnectionMatrix.append(mat)
			prevSize = config.hiddenLayerSize

		if useDynamicGeneratedHiddenConnections:
			# self.register_buffer('neuronSegmentAssignedMask', torch.zeros(config.numberOfHiddenLayers, config.hiddenLayerSize, dtype=torch.bool, device=device)) # Reverted
			self.neuronSegmentAssignedMask = torch.zeros(self.numberUniqueLayers, config.hiddenLayerSize, numberOfSegmentsPerNeuron, dtype=torch.bool, device=device) # Ensure device, Added numberOfSegmentsPerNeuron, Modified

		# -----------------------------
		# Output connection matrix
		# -----------------------------
		if useOutputConnectionsLastLayer:
			outConnShape = (config.hiddenLayerSize, config.numberOfClasses,)
		else:
			outConnShape = (self.numberUniqueLayers, config.hiddenLayerSize, config.numberOfClasses,) # Modified
		if useBinaryOutputConnections:
			self.outputConnectionMatrix = torch.zeros(outConnShape, dtype=torch.bool, device=device) # Added device=device
		else:
			print("outConnShape = ", outConnShape)
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
			S = numberOfSegmentsPerNeuron # Added
			'''
			self.hiddenHashes       = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
			if useEIneurons:
				self.hiddenHashesExc = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
				self.hiddenHashesInh = [[torch.empty(0, dtype=torch.int64, device=device) for _ in range(S)] for _ in range(L)] # Modified
			'''
			self.hiddenNeuronSignatures = [[dict() for _ in range(S)] for _ in range(L+1)]
			if useEIneurons:
				self.hiddenNeuronSignaturesExc = [[dict() for _ in range(S)] for _ in range(L+1)]
				self.hiddenNeuronSignaturesInh = [[dict() for _ in range(S)] for _ in range(L+1)]
		

	# ---------------------------------------------------------
	# Helper - layer initialisation
	# ---------------------------------------------------------

	def getUniqueLayerIndex(self, layerIdSuperblock: int, layerIdHidden: int) -> int: # Added
		if(recursiveLayers):
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

		- If `useDynamicGeneratedHiddenConnections` is True  -> start empty
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
		s = numberOfSegmentsPerNeuron # shorthand, Added
		nnz_per_segment = k # numNeurons * k # Modified: nnz is per segment now for sparse
		nnz = numNeurons * s * k # Total non-zero elements if all segments initialized

		if useSparseMatrix:

			sparse_dtype = torch.bool
			if(initialiseSANIlayerWeightsUsingCPU):
				devTemp =  pt.device('cpu')
			else:
				devTemp = dev
				
			if useDynamicGeneratedHiddenConnections:
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

				if useEIneurons:
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

			if not useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell segment
				for n in range(numNeurons):
					for seg_idx in range(s): # Added loop for segments
						syn_idx = torch.randperm(prevSize, device=dev)[:k]

						if useEIneurons:
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
		#print("device = ", device)

		if useTabularDataset:
			encoded = self._encodeContinuousVarsAsBits(x)
			prevActivation = encoded.to(torch.int8)
		elif useImageDataset:
			encoded = self._encodeContinuousVarsAsBits(x)
			prevActivation = EISANIpt_EISANImodelCNN._propagate_conv_layers(self, encoded)	# (batch, encodedFeatureSize) int8
		elif useNLPDataset:
			print("x.shape = ", x.shape)
			print("x = ", x)
			embedding = EISANIpt_EISANImodelNLP.encodeTokensInputIDs(self, x)	# (batch, sequenceLength, embeddingSize) float32
			print("embedding.shape = ", embedding.shape)
			encoded = self._encodeContinuousVarsAsBits(embedding)	#int8	#[batchSize, sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
			print("encoded.shape = ", encoded.shape)
			prevActivation = encoded.to(torch.int8)
			#encodedFeatureSize = EISANIpt_EISANImodelNLP.getEncodedFeatureSize()	#sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
				
		# -----------------------------
		# Pass through hidden layers
		# -----------------------------
		layerActivations: List[torch.Tensor] = []

		for layerIdSuperblock in range(recursiveSuperblocksNumber): # Modified
			for layerIdHidden in range(self.config.numberOfHiddenLayers): # Modified
				uniqueLayerIndex = self.getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
				if useEIneurons:
					aExc, aInh = self._compute_layer_EI(uniqueLayerIndex, prevActivation, device) # Modified
					currentActivation = torch.cat([aExc, aInh], dim=1)
				else:
					currentActivation = self._compute_layer_standard(uniqueLayerIndex, prevActivation, device) # Modified

				layerActivations.append(currentActivation)

				# -------------------------
				# Dynamic hidden connection growth
				# -------------------------
				if (trainOrTest and useDynamicGeneratedHiddenConnections):
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

		if useOutputConnectionsLastLayer:
			lastLayerActivation = layerActivations[-1] # Activations from the last hidden layer
			
			if useBinaryOutputConnectionsEffective:
				weights = self.outputConnectionMatrix.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
			else:
				if useBinaryOutputConnections:
					weights = self.outputConnectionMatrix.to(torch.int8) # bool to int8 (0/1)
				else:
					weights = self.outputConnectionMatrix # float
					
			weights = self.normaliseOutputConnectionWeights(weights.float())	# cast weights to float for matmul
			outputActivations += lastLayerActivation.float() @ weights	# cast act to float for matmul
			
			if trainOrTest and y is not None:
				# For the last layer, layerIdx is effectively self.config.numberOfHiddenLayers - 1
				# but since outputConnectionMatrix is 2D, we don't pass layerIdx or pass a dummy one if the function expects it
				#TODO: this is probably wrong for recursive layers; should be uniqueLayerIndex?
				# Determine the correct uniqueLayerIndex for the last layer in the last superblock
				lastSuperblockIndex = recursiveSuperblocksNumber - 1
				lastHiddenLayerInSuperblockIndex = self.config.numberOfHiddenLayers -1
				uniqueLayerIndexForLastLayer = self.getUniqueLayerIndex(lastSuperblockIndex, lastHiddenLayerInSuperblockIndex)
				self._update_output_connections(uniqueLayerIndexForLastLayer, lastLayerActivation, y, device)
		else:
			actLayerIndex = 0 # Added
			for layerIdSuperblock in range(recursiveSuperblocksNumber): # Added
				for layerIdHidden in range(self.config.numberOfHiddenLayers): # Added
					act = layerActivations[actLayerIndex] # Modified
					uniqueLayerIndex = self.getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
					weights = self.outputConnectionMatrix[uniqueLayerIndex] # Modified
					
					if useBinaryOutputConnectionsEffective:
						weights = weights.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
					else:
						if useBinaryOutputConnections:
							weights = weights.to(torch.int8) # bool to int8 (0/1)
						else:
							weights = weights
						
					weights = self.normaliseOutputConnectionWeights(weights.float())	# cast weights to float for matmul
					outputActivations += act.float() @ weights 	# cast act to float for matmul

					# Training: reinforce output connections
					if trainOrTest and y is not None:
						self._update_output_connections(uniqueLayerIndex, act, y, device) # Modified
					actLayerIndex += 1 # Added

		predictions = torch.argmax(outputActivations, dim=1)

		# -----------------------------------------------------------------
		# Update hidden-neuron accuracy statistics (soft-max vote)
		# -----------------------------------------------------------------
		if(limitOutputConnectionsBasedOnAccuracy):
			if y is not None:												# labels available
				#note if useOutputConnectionsLastLayer, only the last index in hiddenNeuronPredictionAccuracy will be valid
				
				# 1) normalise & soft-max output weights
				w_norm  = self.normaliseOutputConnectionWeights(self.outputConnectionMatrix)
				w_soft  = torch.softmax(w_norm, dim=-1)						# same shape as outputConnectionMatrix

				if useOutputConnectionsLastLayer:
					soft_layer_all = w_soft									# [H,C]
					conn_layer_all = (self.outputConnectionMatrix != 0).any(dim=1)	# [H]
				else:
					soft_layer_all = w_soft									# [L,H,C]
					conn_layer_all = (self.outputConnectionMatrix != 0).any(dim=2)	# [L,H]

				for lidx, act in enumerate(layerActivations):
					active_mask = (act != 0)								# [B,H]

					if useOutputConnectionsLastLayer:
						soft_layer = soft_layer_all							# [H,C]
						conn_layer = conn_layer_all							# [H]
					else:
						soft_layer = soft_layer_all[lidx]					# [H,C]
						conn_layer = conn_layer_all[lidx]					# [H]

					# 2) gather the soft-max weight for the true class of each sample
					#    soft_layer[:, y] -> [H,B] -> transpose -> [B,H]
					soft_true = soft_layer[:, y].t()						# [B,H]

					valid_neuron_mask = conn_layer.unsqueeze(0)				# [1,H]
					pred_above_thr    = soft_true > limitOutputConnectionsSoftmaxWeightMin

					correct_neuron = active_mask & valid_neuron_mask & pred_above_thr
					considered     = active_mask & valid_neuron_mask

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

	def normaliseOutputConnectionWeights(self, weights):
		if(useOutputConnectionsNormalised):
			weights = torch.tanh(weights/useOutputConnectionsNormalisationRange)
		else:
			weights = weights
		return weights
		
	def _encodeContinuousVarsAsBits(self, x: torch.Tensor) -> torch.Tensor:
		if(useTabularDataset):
			numBits = EISANITABcontinuousVarEncodingNumBits
		if(useImageDataset):
			numBits = EISANICNNcontinuousVarEncodingNumBits
			if numBits == 1:
				return x
			B, C, H, W = x.shape
			x = x.view(B, C * H * W)  # Flatten pixel dimensions
		elif(useNLPDataset):
			B, L, E = x.shape
			x = x.view(B, L * E)
			numBits = EISANINLPcontinuousVarEncodingNumBits

		if useGrayCode:
			encoded_bits_list = self._gray_code_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		else:
			encoded_bits_list = self._thermometer_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)

		if(useTabularDataset):
			code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension	#[B, nCont*EISANITABcontinuousVarEncodingNumBits]
		elif(useImageDataset):
			code = torch.stack(encoded_bits_list, dim=2) 	#[B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
			code = code.view(B, C, H, W, numBits)	#unflatten pixel dimensions
			code = code.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, C, EISANICNNcontinuousVarEncodingNumBits, H, W]
			code = code.reshape(B, C*numBits, H, W)
		elif(useNLPDataset):
			code = torch.stack(encoded_bits_list, dim=1) 	#[B, nCont*EISANINLPcontinuousVarEncodingNumBits]
			#code = code.reshape(B, L, E*numBits)		#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
			code = code.reshape(B, L*E*numBits)

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
		
		B = prevActivation.shape[0]
		numExcNeurons, numSegments, _ = wExc_3d.shape
		
		if useInhibition and self.hiddenConnectionMatrixInhibitory[layerIdx].numel() > 0:
			wInh_3d = self.hiddenConnectionMatrixInhibitory[layerIdx].to(dev) # [numInhNeurons, numSegments, prevSize]
			numInhNeurons, _, _ = wInh_3d.shape
		else:
			wInh_3d = None
			numInhNeurons = 0

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
		if useInhibition and wInh_3d is not None:
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
		else:
			# No inhibitory neurons when inhibition is disabled
			aInh = torch.zeros(B, 0, dtype=torch.int8, device=dev)
		
		return aExc, aInh

	def _segmentActivationFunction(self, z_all_segments):
		# z_all_segments has shape [B, numNeurons, numSegments]
		# A segment fires if its activation sum meets the threshold.
		segment_fires = z_all_segments >= segmentActivationThreshold # [B, numNeurons, numSegments] (bool)
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
			if useOutputConnectionsLastLayer:
				if useBinaryOutputConnections:
					self.outputConnectionMatrix[activeNeurons, targetClass] = True
				else:
					self.outputConnectionMatrix[activeNeurons, targetClass] += 1.0
			else:
				if useBinaryOutputConnections:
					self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] = True
				else:
					self.outputConnectionMatrix[layerIdx, activeNeurons, targetClass] += 1.0

	def executePostTrainPrune(self, trainOrTest):
		EISANIpt_EISANImodelPrune.executePostTrainPrune(self, trainOrTest)


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
	import torch.nn.functional as F
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
		self.hiddenConnectionMatrix: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueLayers)]
		self.hiddenConnectionMatrixExcitatory: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueLayers)]
		self.hiddenConnectionMatrixInhibitory: List[List[torch.Tensor]] = [[] for _ in range(self.numberUniqueLayers)]

		for layerIdx in range(self.numberUniqueLayers): # Modified
			for segmentIdx in range(numberOfSegmentsPerNeuron):
				if useEIneurons: # Corrected: use useEIneurons
					if useInhibition:
						excitSize = config.hiddenLayerSize // 2
						inhibSize = config.hiddenLayerSize - excitSize
					else:
						excitSize = config.hiddenLayerSize  # All neurons are excitatory
						inhibSize = 0
					excMat = self._initialise_layer_weights(excitSize, prevSize, layerIdx) # Corrected: added layerIdx
					self.hiddenConnectionMatrixExcitatory[layerIdx].append(excMat)
					if useInhibition and inhibSize > 0:
						inhMat = self._initialise_layer_weights(inhibSize, prevSize, layerIdx) # Corrected: added layerIdx
						self.hiddenConnectionMatrixInhibitory[layerIdx].append(inhMat)
					else:
						# Create empty inhibitory matrix when inhibition is disabled
						self.hiddenConnectionMatrixInhibitory[layerIdx].append(torch.empty(0, prevSize, device=device))
				else:
					mat = self._initialise_layer_weights(config.hiddenLayerSize, prevSize, layerIdx) # Corrected: added layerIdx
					self.hiddenConnectionMatrix[layerIdx].append(mat)
			prevSize = config.hiddenLayerSize

		if(useSequentialSANI):
			self.layerActivations = torch.zeros(self.numberUniqueLayers, config.hiddenLayerSize, dtype=torch.bool, device=device)
			#self.layerSegmentActivations = torch.zeros(self.numberUniqueLayers, numberOfSegmentsPerNeuron, config.hiddenLayerSize, dtype=torch.bool, device=device)	#not currently used
			self.lastActivationTime = torch.zeros(self.numberUniqueLayers, config.hiddenLayerSize, dtype=torch.int, device=device)
				
		if useDynamicGeneratedHiddenConnections:
			self.neuronSegmentAssignedMask = torch.zeros(self.numberUniqueLayers, numberOfSegmentsPerNeuron, config.hiddenLayerSize, dtype=torch.bool, device=device) # Ensure device, Added numberOfSegmentsPerNeuron, Modified

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
		nnz  = numNeurons * k
    
		if useSparseMatrix:

			sparse_dtype = torch.bool
			if(initialiseSANIlayerWeightsUsingCPU):
				devTemp =  pt.device('cpu')
			else:
				devTemp = dev
				
			if useDynamicGeneratedHiddenConnections:
				# start EMPTY
				indices = torch.empty((2, 0), dtype=torch.int64, device=devTemp)
				values  = torch.empty((0,),  dtype=sparse_dtype, device=devTemp)
			else:
				# start with k random synapses per neuron
				'''
				row_idx = torch.arange(numNeurons, device=dev).repeat_interleave(k)
				col_idx = torch.randint(prevSize, (numNeurons * k,), device=dev)
				indices = torch.stack([row_idx, col_idx])  # shape [2, nnz]
				'''
				# ---------- memory-tight random initialisation ----------
				indices = torch.empty((2, nnz), dtype=torch.int64, device=devTemp)
				# 1. columns: generate in-place
				torch.randint(prevSize, (nnz,), dtype=torch.int64, device=devTemp, out=indices[1])
				# 2. rows: broadcast one small vector into the big slot buffer
				rows = torch.arange(numNeurons, dtype=torch.int64, device=devTemp)  # length = numNeurons
				indices[0].view(numNeurons, k).copy_(rows.unsqueeze(1))		 # expand, copy once

				if useEIneurons:
					values = torch.ones(nnz, device=devTemp, dtype=sparse_dtype)
				else:
					values = torch.randint(0, 2, (nnz,), device=devTemp, dtype=sparse_dtype)
         
			mat = torch.sparse_coo_tensor(indices, values, size=(numNeurons, prevSize), device=devTemp, dtype=sparse_dtype,).coalesce()
			mat = mat.to(dev)
			return mat
		else:
			# -------------------------------------------------- dense initialisation
			weight = torch.zeros(numNeurons, prevSize, device=dev, dtype=torch.int8) # Use torch.int8 for dense

			if not useDynamicGeneratedHiddenConnections:
				# randomly choose k unique presynaptic neurons per postsynaptic cell
				for n in range(numNeurons):
					syn_idx = torch.randperm(prevSize, device=dev)[:k]

					if useEIneurons:
						weight[n, syn_idx] = 1
					else:
						# Generate +1 or -1 with equal probability, as int8
						rand_signs_bool = torch.randint(0, 2, (k,), device=dev, dtype=torch.bool)
						rand_signs = torch.where(rand_signs_bool, torch.tensor(1, device=dev, dtype=torch.int8), torch.tensor(-1, device=dev, dtype=torch.int8))
						weight[n, syn_idx] = rand_signs

			# make it a learnable parameter for the dense case
			return weight

	# ---------------------------------------------------------
	# Forward pass
	# ---------------------------------------------------------

	@torch.no_grad()
	def forward(self, trainOrTest: bool, x: torch.Tensor, y: Optional[torch.Tensor] = None, optim=None, l=None, batchIndex=None, fieldTypeList=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

		if(useNLPDataset):
			if(useNeuronActivationMemory):
				assert batchSize == 1
				numSubsamples = sequenceLength
				seq = x[0]               # Tensor [sequenceLength]
				non_pad = (seq != NLPcharacterInputPadTokenID)
				if not pt.any(non_pad):
					return
				max_idx = pt.nonzero(non_pad, as_tuple=True)[0][-1].item()
				numSubsamples = max_idx+1
				seqReal = seq[: max_idx + 1]
				extra = contextSizeMax - max_idx      # spec guarantees > 0
				print("numSubsamples = ", numSubsamples)
			else:
				numSubsamples = 1
		else:
			numSubsamples = 1
			
		for subsampleIndex in range(numSubsamples):

			# -----------------------------
			# Apply sliding window (sequence input only)
			# -----------------------------
			if(useNLPDataset and useNeuronActivationMemory):
				shift = subsampleIndex + extra
				x_shift  = pt.full((contextSizeMax,), NLPcharacterInputPadTokenID, dtype=seq.dtype, device=seq.device)
				copy_len = min(seqReal.size(0), contextSizeMax - shift)
				if copy_len > 0:
					x_shift[shift : shift + copy_len] = seqReal[: copy_len]
				y_shift = x_shift[-1]
				x = x_shift.unsqueeze(0)	#add redundant batch dim
				y = y_shift.unsqueeze(0)	#add redundant batch dim
				
			# -----------------------------
			# Continuous var encoding as bits
			# -----------------------------
			initActivation = self._continuousVarEncoding(x)
			
			# -----------------------------
			# Hidden layers
			# -----------------------------
			if(useSequentialSANI):
				layerActivations = EISANIpt_EISANImodelNLP._sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, initActivation)
			else:
				layerActivations = self._summationSANIpassHiddenLayers(trainOrTest, initActivation)

			# -----------------------------
			# Output layer
			# -----------------------------
			predictions = self._calculateOutputLayer(trainOrTest, layerActivations, y)

			# -----------------------------------------------------------------
			# Update hidden-neuron accuracy statistics (soft-max vote)
			# -----------------------------------------------------------------
			self._updateHiddenNeuronAccuracyStatistics()
			
			# count how many are exactly correct
			correct = (predictions == y).sum().item()
			accuracy = correct / y.size(0)
			loss = Loss(0.0)
		
		return loss, accuracy

	# -------------------------------------------------------------
	# Continuous var encoding as bits
	# -------------------------------------------------------------

	def _continuousVarEncoding(self, x):
		if useTabularDataset:
			encoded = self._encodeContinuousVarsAsBits(x)
			initActivation = encoded.to(torch.int8)
			numSubsamples = 1
		elif useImageDataset:
			encoded = self._encodeContinuousVarsAsBits(x)
			initActivation = EISANIpt_EISANImodelCNN._propagate_conv_layers(self, encoded)	# (batch, encodedFeatureSize) int8
		elif useNLPDataset:
			print("x.shape = ", x.shape)
			print("x = ", x)
			embedding = EISANIpt_EISANImodelNLP.encodeTokensInputIDs(self, x)	# (batch, sequenceLength, embeddingSize) float32
			#print("embedding.shape = ", embedding.shape)
			encoded = self._encodeContinuousVarsAsBits(embedding)	#int8	#[batchSize, sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
			#print("encoded.shape = ", encoded.shape)
			#print("encoded = ", encoded)
			initActivation = encoded.to(torch.int8)
			#encodedFeatureSize = EISANIpt_EISANImodelNLP.getEncodedFeatureSize()	#sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
		return initActivation

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
			numBits = EISANINLPcontinuousVarEncodingNumBits
			if(useNLPcharacterInput):
				B, L = x.shape
				x = x.view(B, L)
			else:
				B, L, E = x.shape
				x = x.view(B, L * E)

		if useContinuousVarEncodeMethod=="grayCode":
			encoded_bits_list = self._gray_code_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		elif useContinuousVarEncodeMethod=="thermometer":
			encoded_bits_list = self._thermometer_encode(x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		elif useContinuousVarEncodeMethod=="onehot":
			encoded_bits = F.one_hot(x, num_classes=numBits)	# (B, L, numBits)

		if(useTabularDataset):
			code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension	#[B, nCont*numBits]
		elif(useImageDataset):
			code = torch.stack(encoded_bits_list, dim=2) 	#[B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
			code = code.view(B, C, H, W, numBits)	#unflatten pixel dimensions
			code = code.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, C, numBits, H, W]
			code = code.reshape(B, C*numBits, H, W)
		elif(useNLPDataset):
			if useContinuousVarEncodeMethod=="onehot":
				code = encoded_bits.view(B, L, numBits) 	#[B, L*numBits]
				#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, EISANINLPcontinuousVarEncodingNumBits]
				code = code.reshape(B, L*numBits)
			else:
				code = torch.stack(encoded_bits_list, dim=1) 	#[B, L, numBits]
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
		
	# -----------------------------
	# Hidden layers
	# -----------------------------
	
	def _summationSANIpassHiddenLayers(self, trainOrTest, initActivation):
		prevActivation = initActivation
		layerActivations: List[torch.Tensor] = []
		for layerIdSuperblock in range(recursiveSuperblocksNumber): # Modified
			for layerIdHidden in range(self.config.numberOfHiddenLayers): # Modified
				uniqueLayerIndex = self.getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
				if useEIneurons:
					aExc, aInh = self._compute_layer_EI(uniqueLayerIndex, prevActivation, device) # Modified
					currentActivation = torch.cat([aExc, aInh], dim=1)
				else:
					currentActivation = self._compute_layer_standard(uniqueLayerIndex, prevActivation, device) # Modified

				# -------------------------
				# Dynamic hidden connection growth
				# -------------------------
				if (trainOrTest and useDynamicGeneratedHiddenConnections):
					for _ in range(numberNeuronSegmentsGeneratedPerSample):
						if(useDynamicGeneratedHiddenConnectionsVectorised):
							EISANIpt_EISANImodelDynamic.dynamic_hidden_growth_vectorised(self, uniqueLayerIndex, prevActivation, currentActivation, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified
						else:
							for s_batch_idx in range(prevActivation.size(0)):                # loop over batch
								prevAct_b  = prevActivation[s_batch_idx : s_batch_idx + 1]             # keep 2- [1, prevSize]
								currAct_b  = currentActivation[s_batch_idx : s_batch_idx + 1]          # keep 2- [1, layerSize]
								EISANIpt_EISANImodelDynamic.dynamic_hidden_growth(self, uniqueLayerIndex, prevAct_b, currAct_b, device, segmentIndexToUpdate) # Added segmentIndexToUpdate, Modified

				layerActivations.append(currentActivation)
				prevActivation = currentActivation
		return layerActivations

	def _compute_layer_standard(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> torch.Tensor:
		activatedAllSegments = []
		
		for segmentIdx in range(numberOfSegmentsPerNeuron):
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
			activated = self._segmentActivationFunction(z_float).to(torch.int8) # bool to int8 (0 or 1)
			activatedAllSegments.append(activated)
		activated = self._neuronActivationFunction(activatedAllSegments)
		return activated

	def _compute_layer_EI(self, layerIdx: int, prevActivation: torch.Tensor, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
		aExcAllSegments = []
		aInhAllSegments = []
		for segmentIdx in range(numberOfSegmentsPerNeuron):
			# prevActivation is torch.int8 (0 or 1)
			dev  = prevActivation.device
			wExc = self.hiddenConnectionMatrixExcitatory[layerIdx][segmentIdx].to(dev)
			wInh = self.hiddenConnectionMatrixInhibitory[layerIdx][segmentIdx].to(dev)

			# Excitatory
			if useSparseMatrix:
				# EI sparse weights are True for +1
				numeric_values_exc_float = wExc.values().to(torch.float32) # True becomes 1.0
				wExc_eff_float = torch.sparse_coo_tensor(wExc.indices(), numeric_values_exc_float, wExc.shape, device=dev, dtype=torch.float32).coalesce()
				zExc_float = torch.sparse.mm(wExc_eff_float, prevActivation.float().t()).t()
			else: # dense
				# Dense EI weights are 1 (int8). Convert to float for matmul.
				zExc_float = prevActivation.float() @ wExc.float().t()
			aExc = self._segmentActivationFunction(zExc_float).to(torch.int8) # bool to int8 (0 or 1)
			aExcAllSegments.append(aExc)
			
			# Inhibitory
			if useSparseMatrix:
				# EI sparse weights are True for +1
				numeric_values_inh_float = wInh.values().to(torch.float32) # True becomes 1.0
				wInh_eff_float = torch.sparse_coo_tensor(wInh.indices(), numeric_values_inh_float, wInh.shape, device=dev, dtype=torch.float32).coalesce()
				zInh_float = torch.sparse.mm(wInh_eff_float, prevActivation.float().t()).t()
			else: # dense
				# Dense EI weights are 1 (int8). Convert to float for matmul.
				zInh_float = prevActivation.float() @ wInh.float().t()
			firesInh = self._segmentActivationFunction(zInh_float)
			aInh = torch.zeros_like(zInh_float, dtype=torch.int8, device=dev) # Initialize with correct shape, device and int8 type
			aInh[firesInh] = -1
			aInhAllSegments.append(aInh)
		aExc = self._neuronActivationFunction(aExcAllSegments)
		aInh = self._neuronActivationFunction(aInhAllSegments)
		return aExc, aInh
		
	def _segmentActivationFunction(self, z_all_segments):
		# z_all_segments has shape [B, numNeurons, numberOfSegmentsPerNeuron]
		# A segment fires if its activation sum meets the threshold.
		segment_fires = z_all_segments >= segmentActivationThreshold # [B, numNeurons, numberOfSegmentsPerNeuron] (bool)
		return segment_fires
		
	def _neuronActivationFunction(self, a_all_segments_list):
		#for EISANI summation activated neuronal input assume neuron is activated when any segments are activated
		a_all_segments = torch.stack(a_all_segments_list, dim=2)	# [B, numNeurons, numberOfSegmentsPerNeuron] (bool)
		# Combine segment activations: a neuron is active if any of its segments are active.
		neuron_fires = torch.any(a_all_segments, dim=2) # [B, numNeurons] (bool)
		return neuron_fires
		
	# -----------------------------
	# Output layer
	# -----------------------------
							
	def _calculateOutputLayer(self, trainOrTest, layerActivations, y):
		outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

		if useOutputConnectionsLastLayer:
			act = layerActivations[-1] # Activations from the last hidden layer
			weights = self.outputConnectionMatrix

			if useBinaryOutputConnectionsEffective:
				weights = self.outputConnectionMatrix.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
			else:
				if useBinaryOutputConnections:
					weights = self.outputConnectionMatrix.to(torch.int8) # bool to int8 (0/1)
				else:
					weights = self.outputConnectionMatrix # float

			weights = self._normaliseOutputConnectionWeights(weights.float())	# cast weights to float for matmul
			outputActivations += act.float() @ weights	# cast act to float for matmul

			if trainOrTest and y is not None:
				self._update_output_connections(None, act, y, device)
		else:
			actLayerIndex = 0
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

					weights = self._normaliseOutputConnectionWeights(weights.float())	# cast weights to float for matmul
					outputActivations += act.float() @ weights 	# cast act to float for matmul

					# Training: reinforce output connections
					if trainOrTest and y is not None:
						self._update_output_connections(uniqueLayerIndex, act, y, device)
					actLayerIndex += 1 # Added

		predictions = torch.argmax(outputActivations, dim=1)
		return predictions
		
	def _normaliseOutputConnectionWeights(self, weights):
		if(useOutputConnectionsNormalised):
			weights = torch.tanh(weights/useOutputConnectionsNormalisationRange)
		else:
			weights = weights
		return weights
	
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
						
	# -----------------------------------------------------------------
	# Update hidden-neuron accuracy statistics (soft-max vote)
	# -----------------------------------------------------------------
				
	def _updateHiddenNeuronAccuracyStatistics(self):
		if(limitOutputConnectionsBasedOnAccuracy):
			if y is not None:												# labels available
				#note if useOutputConnectionsLastLayer, only the last index in hiddenNeuronPredictionAccuracy will be valid

				# 1) normalise & soft-max output weights
				w_norm  = self._normaliseOutputConnectionWeights(self.outputConnectionMatrix)
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

		
	def executePostTrainPrune(self, trainOrTest):
		EISANIpt_EISANImodelPrune.executePostTrainPrune(self, trainOrTest)


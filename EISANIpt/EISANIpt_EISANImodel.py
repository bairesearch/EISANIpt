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

"""

import torch
from torch import nn
from typing import List, Optional, Tuple
import EISANIpt_EISANI_globalDefs
from ANNpt_globalDefs import *
import EISANIpt_EISANImodelContinuousVarEncoding
if(useTabularDataset):
	import EISANIpt_EISANImodelSummation
elif(useImageDataset):
	import EISANIpt_EISANImodelCNN
	import EISANIpt_EISANImodelSummation
elif(useNLPDataset):
	import EISANIpt_EISANImodelNLP
	import EISANIpt_EISANImodelSequential
import EISANIpt_EISANImodelOutput
if(limitConnections):
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

def getNumberUniqueHiddenLayers(recursiveLayers, recursiveSuperblocksNumber, numberOfHiddenLayers):
	if(recursiveLayers):
		numberUniqueHiddenLayers = recursiveSuperblocksNumber*2
		#*2 explanation: the first forward propagated layer in a superblock always uses unique weights:
		#	- for the first superblock this will comprise unique weights between the input layer and the first hidden layer of the superblock
		#	- for the all other superblocks this will comprise unique weights between the previous superblock and the first hidden layer of the superblock
	else:
		numberUniqueHiddenLayers = numberOfHiddenLayers
	return numberUniqueHiddenLayers

def generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers, hiddenLayerSize):
	numberOfHiddenLayers = generateNumberHiddenLayers(numberOfLayers, numberOfConvlayers)
	if(useSequentialSANI):
		hiddenLayerSizeSANI = -1	#hidden layers are dynamically sized [input layer == encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits]
	else:
		if(useStochasticUpdates):
			hiddenLayerSizeSANI = hiddenLayerSize*hiddenLayerSizeSANImultiplier
		else:
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

		self.numberUniqueHiddenLayers = getNumberUniqueHiddenLayers(recursiveLayers, recursiveSuperblocksNumber, config.numberOfHiddenLayers)

		if(useSequentialSANI):
			if(not evalOnlyUsingTimeInvariance):
				self.sequentialSANItimeInvariance = sequentialSANItimeInvariance
				self.useSequentialSANIactivationStrength = useSequentialSANIactivationStrength
				
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

		if(useConnectionWeights):
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
						excMat = self._initialise_layer_weights(excitSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
						self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx].append(excMat)
						if useInhibition and inhibSize > 0:
							inhMat = self._initialise_layer_weights(inhibSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
							self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(inhMat)
						else:
							# Create empty inhibitory matrix when inhibition is disabled
							self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(torch.empty(0, prevSize, device=device))
					else:
						mat = self._initialise_layer_weights(config.hiddenLayerSize, prevSize, hiddenLayerIdx) # Corrected: added hiddenLayerIdx
						self.hiddenConnectionMatrix[hiddenLayerIdx].append(mat)
				prevSize = config.hiddenLayerSize

			if useDynamicGeneratedHiddenConnections:
				self.neuronSegmentAssignedMask = torch.zeros(self.numberUniqueHiddenLayers, numberOfSegmentsPerNeuron, config.hiddenLayerSize, dtype=torch.bool, device=device) # Ensure device, Added numberOfSegmentsPerNeuron, Modified

			# -----------------------------
			# verify neuron uniqueness
			# -----------------------------
			if(useDynamicGeneratedHiddenConnections and useDynamicGeneratedHiddenConnectionsUniquenessChecks):
				L = self.numberUniqueHiddenLayers # Modified
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

		if(useConnectionWeights):
			hiddenLayerSizeStart = config.hiddenLayerSize
		else:
			hiddenLayerSizeStart = blockInitCapacity
	
		if(useSequentialSANI):
			self.numAssignedNeuronSegments = torch.zeros(self.numberUniqueHiddenLayers, dtype=torch.long, device=device)
			self.indexArrayA: List[torch.Tensor] = [torch.full((hiddenLayerSizeStart,), -1, dtype=torch.long, device=device) for _ in range(self.numberUniqueHiddenLayers)]
			self.indexArrayB: List[torch.Tensor] = [torch.full((hiddenLayerSizeStart,), -1, dtype=torch.long, device=device) for _ in range(self.numberUniqueHiddenLayers)]
	
		# -----------------------------
		# Output connection matrix
		# -----------------------------
		outConnShape = (hiddenLayerSizeStart, config.numberOfClasses,) # Modified
		dtype_out = torch.bool if useBinaryOutputConnections else torch.float32
		if(useSparseOutputMatrix):
			#empty_crow = torch.tensor([0], device=device, dtype=torch.int64)
			rows = outConnShape[0]
			empty_crow = torch.zeros(rows + 1, device=device, dtype=torch.int64)
			empty_col  = torch.tensor([], device=device, dtype=torch.int64)
			empty_val  = torch.tensor([], device=device, dtype=dtype_out)
			if(useOutputConnectionsLastLayer):
				self.outputConnectionMatrix = [None for _ in range(self.numberUniqueHiddenLayers)]
				self.outputConnectionMatrix[self.numberUniqueHiddenLayers-1] = torch.sparse_csr_tensor(empty_crow, empty_col, empty_val, size=outConnShape, device=device)
			else:
				self.outputConnectionMatrix: List[torch.Tensor] = [torch.sparse_csr_tensor(empty_crow, empty_col, empty_val, size=outConnShape, device=device) for _ in range(self.numberUniqueHiddenLayers)]
		else:
			if(useOutputConnectionsLastLayer):
				self.outputConnectionMatrix = [None for _ in range(self.numberUniqueHiddenLayers)]
				self.outputConnectionMatrix[self.numberUniqueHiddenLayers-1] = torch.zeros(outConnShape, dtype=dtype_out, device=device)
			else:
				self.outputConnectionMatrix: List[torch.Tensor] = [torch.zeros(outConnShape, dtype=dtype_out, device=device) for _ in range(self.numberUniqueHiddenLayers)]

			# -------------------------------------------------------------
			# Stochastic mode: dense, fully-connected outputs with random init
			# -------------------------------------------------------------
			if useStochasticUpdates:
				def _init_dense_output():
					W = torch.empty(outConnShape, dtype=torch.float32, device=device)
					torch.nn.init.uniform_(W, -0.01, 0.01)
					return W
				if(useOutputConnectionsLastLayer):
					self.outputConnectionMatrix = [None for _ in range(self.numberUniqueHiddenLayers)]
					self.outputConnectionMatrix[self.numberUniqueHiddenLayers-1] = _init_dense_output()
				else:
					self.outputConnectionMatrix = [_init_dense_output() for _ in range(self.numberUniqueHiddenLayers)]

		# -------------------------------------------------------------
		# limitConnections
		# -------------------------------------------------------------
		if(limitConnections):
			if(limitOutputConnections and limitOutputConnectionsBasedOnAccuracy):
				#Hidden-neuron prediction-accuracy tracker; dim-2: 0 -> #correct, 1 -> #total
				self.hiddenNeuronPredictionAccuracy: List[torch.Tensor] = [torch.zeros((hiddenLayerSizeStart, 2), dtype=torch.int, device=device) for _ in range(self.numberUniqueHiddenLayers)]
			self.hiddenNeuronUsage: List[torch.Tensor] = [torch.zeros((hiddenLayerSizeStart,), dtype=torch.float, device=device) for _ in range(self.numberUniqueHiddenLayers)]

		
	# ---------------------------------------------------------
	# Layer initialisation
	# ---------------------------------------------------------
	
	if(useSequentialSANI):
		def _setTimeInvarianceState(self, trainOrTest):
			if(evalOnlyUsingTimeInvariance):
				if(trainOrTest):
					self.sequentialSANItimeInvariance = False
					self.useSequentialSANIactivationStrength = False
				else:
					self.sequentialSANItimeInvariance = sequentialSANItimeInvariance
					self.useSequentialSANIactivationStrength = useSequentialSANIactivationStrength
					
		def _initialiseLayerActivations(self):
			numberUniqueLayers = self.numberUniqueHiddenLayers+1	#activation arrays contain both input and hidden neurons (not only hidden neurons)
			self.layerActivation: List[torch.Tensor] = [torch.zeros((self.batchSize, self._getCurrentLayerSize(layerIdx),), dtype=torch.bool, device=device) for layerIdx in range(numberUniqueLayers)]	#record of activation state at last activation time (recorded activations will grow over time)
			self.layerActivationTime: List[torch.Tensor] = [torch.zeros((self.batchSize, self._getCurrentLayerSize(layerIdx),), dtype=torch.int, device=device) for layerIdx in range(numberUniqueLayers)]	#last activation time
			if(self.useSequentialSANIactivationStrength):
				self.layerActivationDistance: List[torch.Tensor] = [torch.zeros((self.batchSize, self._getCurrentLayerSize(layerIdx),), dtype=torch.int, device=device) for layerIdx in range(numberUniqueLayers)]	#distance between neuron segments (additive recursive)
				self.layerActivationCount: List[torch.Tensor] = [torch.zeros((self.batchSize, self._getCurrentLayerSize(layerIdx),), dtype=torch.int, device=device) for layerIdx in range(numberUniqueLayers)]		#count number of subneurons which were activated
				#self.layerActivationStrength: List[torch.Tensor] = [torch.zeros((self.batchSize, self._getCurrentLayerSize(layerIdx),), dtype=torch.float, device=device) for layerIdx in range(numberUniqueLayers)]	#not currently used (it is a derived parameter)
	
	def _getCurrentLayerSize(self, layerIdx):
		if(useSequentialSANI):
			if(layerIdx==0):
				#first layer activations use static number abstract neurons (ie neuron activations/times recorded at time t);
				currentLayerSize = self.encodedFeatureSize
			else:
				hiddenLayerIdx = layerIdx-1
				currentLayerSize = self.indexArrayA[hiddenLayerIdx].shape[0]
		else:
			if(layerIdx==0):
				currentLayerSize = self.encodedFeatureSize
			else:
				currentLayerSize = self.config.hiddenLayerSize
		return currentLayerSize

	def _getCurrentHiddenLayerSize(self, hiddenLayerIdx):
		return self._getCurrentLayerSize(hiddenLayerIdx+1)
				
	if(useConnectionWeights):
		def _initialise_layer_weights(self, numNeurons: int, prevSize: int, hiddenLayerIdx: int,) -> torch.Tensor:
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
		self.batchSize = x.shape[0]
		#assert (self.batchSize == self.config.batchSize), "Batch size must match config.batchSize"
		device = x.device

		if(useSlidingWindow):
			# --- updated for full-batch processing ---
			seq	= x	# Tensor [batchSize, sequenceLength]
			non_pad	= (seq != NLPcharacterInputPadTokenID)		# [B, L]
			if not pt.any(non_pad):
				return
			lengths	= non_pad.sum(-1)							# [B]
			max_len = int(lengths.max().item())
			numSubsamples = max(1, max_len - 1)				# predict *next* token only
			extra = contextSizeMax - lengths					# [B]
		else:
			numSubsamples = 1
		
		if(useSequentialSANI):
			self._setTimeInvarianceState(trainOrTest)
			self._initialiseLayerActivations()
		else:
			self.useSequentialSANIactivationStrength = False
		
		accuracyAllWindows = 0
		#print("numSubsamples = ", numSubsamples)
		for slidingWindowIndex in range(numSubsamples):
			if(debugSequentialSANIactivationsLoops):
				print("\n************************** slidingWindowIndex = ", slidingWindowIndex)
			
			# -----------------------------
			# Apply sliding window (sequence input only)
			# -----------------------------
			if(useSlidingWindow):
				# --- one-token sliding window: output shape = [B,1] ---
				token_idx = slidingWindowIndex								# scalar int
				idx = pt.full((self.batchSize, 1), token_idx, dtype=pt.long, device=device)
				gather_idx = idx.clamp(max=seq.size(1) - 1)

				next_idx = pt.full((self.batchSize, 1), token_idx + 1, dtype=pt.long, device=device)
				next_idx_clamped = next_idx.clamp(max=seq.size(1) - 1)

				# current token
				x_shift = seq.gather(1, gather_idx)						# [B,1]
				# next-token target
				y_shift = seq.gather(1, next_idx_clamped).squeeze(1)		# [B]

				# zero-out positions past true length
				invalid_x	= (token_idx >= lengths).unsqueeze(1)			# [B,1]
				invalid_y	= (token_idx + 1 >= lengths)					# [B]
				x_shift[invalid_x] = NLPcharacterInputPadTokenID
				y_shift[invalid_y] = NLPcharacterInputPadTokenID

				x = x_shift											# [B,1]
				y = y_shift											# [B]
				 
			# -----------------------------
			# Continuous var encoding as bits
			# -----------------------------
			initActivation = EISANIpt_EISANImodelContinuousVarEncoding.continuousVarEncoding(self, x)
			
			# -----------------------------
			# Hidden layers
			# -----------------------------
			if(useSequentialSANI):
				layerActivations = EISANIpt_EISANImodelSequential.sequentialSANIpassHiddenLayers(self, trainOrTest, batchIndex, slidingWindowIndex, initActivation)
			else:
				layerActivations = EISANIpt_EISANImodelSummation.summationSANIpassHiddenLayers(self, trainOrTest, initActivation)

			# -----------------------------
			# Output layer
			# -----------------------------
			predictions = EISANIpt_EISANImodelOutput.calculateOutputLayer(self, trainOrTest, layerActivations, y)

			# -----------------------------------------------------------------
			# Update hidden-neuron accuracy statistics (soft-max vote)
			# -----------------------------------------------------------------
			EISANIpt_EISANImodelOutput.updateHiddenNeuronAccuracyStatistics(self, trainOrTest, layerActivations, y)
			
			# count how many are exactly correct
			if(useNLPDataset):
				valid_mask = (y != NLPcharacterInputPadTokenID)
				valid_count = valid_mask.sum().item()
				if valid_count > 0:
					correct = ((predictions == y) & valid_mask).sum().item()
					accuracyAllWindows += correct / valid_count
				else:
					accuracyAllWindows += 0.0
			else:
				correct = (predictions == y).sum().item()
				accuracy = correct / y.size(0)
				accuracyAllWindows += accuracy
		
		accuracy = accuracyAllWindows / numSubsamples
		# Provide a scalar loss usable for stochastic comparison (lower is better)
		loss = Loss(1.0 - float(accuracy))
		
		return loss, accuracy

	def _getUniqueLayerIndex(self, layerIdSuperblock: int, layerIdHidden: int) -> int:
		if(recursiveLayers):
			if(layerIdHidden==0):
				uniqueLayerIndex = layerIdSuperblock*2
			else:
				uniqueLayerIndex = layerIdSuperblock*2+1
		else:
			uniqueLayerIndex = layerIdHidden
		return uniqueLayerIndex


	def executePostTrainPrune(self, trainOrTest):
		EISANIpt_EISANImodelPrune.executePostTrainPrune(self, trainOrTest)

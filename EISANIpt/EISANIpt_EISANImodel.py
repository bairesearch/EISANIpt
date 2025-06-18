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

def generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers):
	numberOfHiddenLayers = generateNumberHiddenLayers(numberOfLayers, numberOfConvlayers)
	if(useSequentialSANI):
		hiddenLayerSizeSANI = -1	#hidden layers are dynamically sized [input layer == encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits]
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
			numberUniqueLayers = self.numberUniqueHiddenLayers+1	#these arrays contain both input and hidden neurons (not only hidden neurons)
			if(useConnectionWeights):
				self.hiddenNeuronPairSignatures = [{} for _ in range(self.numberUniqueHiddenLayers)]
			else:
				self.numAssignedNeuronSegments = torch.zeros(self.numberUniqueHiddenLayers, dtype=torch.long, device=device)
				self.indexArrayA: List[torch.Tensor] = [torch.full((hiddenLayerSizeStart,), -1, dtype=torch.long, device=device) for _ in range(self.numberUniqueHiddenLayers)]
				self.indexArrayB: List[torch.Tensor] = [torch.full((hiddenLayerSizeStart,), -1, dtype=torch.long, device=device) for _ in range(self.numberUniqueHiddenLayers)]
			
			self.layerActivation: List[torch.Tensor] = [torch.zeros((config.batchSize, hiddenLayerSizeStart,), dtype=torch.bool, device=device) for _ in range(numberUniqueLayers)]	#record of activation state at last activation time (recorded activations will grow over time)
			self.layerActivationTime: List[torch.Tensor] = [torch.zeros((config.batchSize, hiddenLayerSizeStart,), dtype=torch.int, device=device) for _ in range(numberUniqueLayers)]	#last activation time
			if(useSequentialSANIactivationStrength):
				self.layerActivationDistance: List[torch.Tensor] = [torch.zeros((config.batchSize, hiddenLayerSizeStart,), dtype=torch.int, device=device) for _ in range(numberUniqueLayers)]	#distance between neuron segments (additive recursive)
				self.layerActivationCount: List[torch.Tensor] = [torch.zeros((config.batchSize, hiddenLayerSizeStart,), dtype=torch.int, device=device) for _ in range(numberUniqueLayers)]		#count number of subneurons which were activated
				#self.layerActivationStrength: List[torch.Tensor] = [torch.zeros((config.batchSize, hiddenLayerSizeStart,), dtype=torch.float, device=device) for _ in range(numberUniqueLayers)]	#not currently used (it is a derived parameter)
			
			#first layer activations use static number abstract neurons (ie neuron activations/times recorded at time t);
			self.layerActivation[0] = torch.zeros((config.batchSize, self.encodedFeatureSize,), dtype=torch.bool, device=device)
			self.layerActivationTime[0] = torch.zeros((config.batchSize, self.encodedFeatureSize,), dtype=torch.int, device=device)
			
		# -----------------------------
		# Output connection matrix
		# -----------------------------
		
		outConnShape = (hiddenLayerSizeStart, config.numberOfClasses,) # Modified
		if(useOutputConnectionsLastLayer):
			self.outputConnectionMatrix = [None for _ in range(self.numberUniqueHiddenLayers)]
			if useBinaryOutputConnections:
				self.outputConnectionMatrix[self.numberUniqueHiddenLayers-1] = torch.zeros(outConnShape, dtype=torch.bool, device=device)
			else:
				self.outputConnectionMatrix[self.numberUniqueHiddenLayers-1] = torch.zeros(outConnShape, dtype=torch.float, device=device)
		else:
			if useBinaryOutputConnections:
				self.outputConnectionMatrix: List[torch.Tensor] = [torch.zeros(outConnShape, dtype=torch.bool, device=device) for _ in range(self.numberUniqueHiddenLayers)]
			else:
				self.outputConnectionMatrix: List[torch.Tensor] = [torch.zeros(outConnShape, dtype=torch.float, device=device) for _ in range(self.numberUniqueHiddenLayers)]

		# -------------------------------------------------------------
		# Hidden-neuron prediction-accuracy tracker
		#   dim-2: 0 -> #correct, 1 -> #total
		# -------------------------------------------------------------
		if(limitOutputConnectionsBasedOnAccuracy):
			self.hiddenNeuronPredictionAccuracy: List[torch.Tensor] = [torch.zeros((hiddenLayerSizeStart, 2), dtype=torch.bool, device=device) for _ in range(self.numberUniqueHiddenLayers)]
			
		
	# ---------------------------------------------------------
	# Helper - layer initialisation
	# ---------------------------------------------------------

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

		if(useNLPDataset):
			if(useNeuronActivationMemory):
				# --- updated for full-batch processing ---
				seq	= x	# Tensor [batchSize, sequenceLength]
				non_pad	= (seq != NLPcharacterInputPadTokenID)		# [B, L]
				if not pt.any(non_pad):
					return
				lengths	= non_pad.sum(-1)							# [B]
				numSubsamples = int(lengths.max().item())			# global max
				extra = contextSizeMax - lengths					# [B]
			else:
				numSubsamples = 1
		else:
			numSubsamples = 1
			
		accuracyAllWindows = 0
		for slidingWindowIndex in range(numSubsamples):
			if(numSubsamples > 1):
				if(debugSequentialSANIactivations):
					print("slidingWindowIndex = ", slidingWindowIndex)
			
			# -----------------------------
			# Apply sliding window (sequence input only)
			# -----------------------------
			if(useNLPDataset and useNeuronActivationMemory):
				# --- one-token sliding window: output shape = [B,1] ---
				token_idx = slidingWindowIndex								# scalar int
				idx = pt.full((batchSize, 1), token_idx, dtype=pt.long, device=device)
				gather_idx = idx.clamp(max=seq.size(1) - 1)					# stay in-bounds

				x_shift = seq.gather(1, gather_idx)						# [B,1]

				# mask out positions beyond each sequence\u2019s true length
				invalid		= (token_idx >= lengths).unsqueeze(1)			# [B,1]
				x_shift[invalid] = NLPcharacterInputPadTokenID

				y_shift = x_shift.squeeze(1)						# [B]
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
			predictions = self._calculateOutputLayer(trainOrTest, layerActivations, y)

			# -----------------------------------------------------------------
			# Update hidden-neuron accuracy statistics (soft-max vote)
			# -----------------------------------------------------------------
			self._updateHiddenNeuronAccuracyStatistics()
			
			# count how many are exactly correct
			correct = (predictions == y).sum().item()
			accuracy = correct / y.size(0)
			accuracyAllWindows += accuracy
			
		accuracyAllWindows = accuracyAllWindows / numSubsamples
		loss = Loss(0.0)
		
		return loss, accuracy

	# -----------------------------
	# Output layer
	# -----------------------------

	def _getUniqueLayerIndex(self, layerIdSuperblock: int, layerIdHidden: int) -> int:
		if(recursiveLayers):
			if(layerIdHidden==0):
				uniqueLayerIndex = layerIdSuperblock*2
			else:
				uniqueLayerIndex = layerIdSuperblock*2+1
		else:
			uniqueLayerIndex = layerIdHidden
		return uniqueLayerIndex

				
	def _calculateOutputLayer(self, trainOrTest, layerActivations, y):
		outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

		actLayerIndex = 0
		for layerIdSuperblock in range(recursiveSuperblocksNumber):
			for layerIdHidden in range(self.config.numberOfHiddenLayers):
				isLastLayer = (layerIdSuperblock==recursiveSuperblocksNumber-1) and (layerIdHidden==self.config.numberOfHiddenLayers-1)
				if(not useOutputConnectionsLastLayer or isLastLayer):
					act = layerActivations[actLayerIndex]
					uniqueLayerIndex = self._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden) # Added
					weights = self.outputConnectionMatrix[uniqueLayerIndex]

					#print("actLayerIndex = ", actLayerIndex)
					#print("act.shape = ", act.shape)
					#print("weights.shape = ", weights.shape)
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
				actLayerIndex += 1

		predictions = torch.argmax(outputActivations, dim=1)
		return predictions
		
	def _normaliseOutputConnectionWeights(self, weights):
		if(useOutputConnectionsNormalised):
			weights = torch.tanh(weights/useOutputConnectionsNormalisationRange)
		else:
			weights = weights
		return weights
	
	def _update_output_connections(self, hiddenLayerIdx: int, activation: torch.Tensor, y: torch.Tensor, device: torch.device,) -> None:
		batchSize = activation.size(0)
		activeMask = activation != 0.0  # (batch, hidden)
		for sampleIdx in range(batchSize):
			targetClass = y[sampleIdx].item()
			activeNeurons = activeMask[sampleIdx].nonzero(as_tuple=True)[0]
			if activeNeurons.numel() == 0:
				continue
			if useBinaryOutputConnections:
				self.outputConnectionMatrix[hiddenLayerIdx][activeNeurons, targetClass] = True
			else:
				self.outputConnectionMatrix[hiddenLayerIdx][activeNeurons, targetClass] += 1.0
						
	# -----------------------------------------------------------------
	# Update hidden-neuron accuracy statistics (soft-max vote)
	# -----------------------------------------------------------------
				
	def _updateHiddenNeuronAccuracyStatistics(self):
		if(limitOutputConnectionsBasedOnAccuracy):
			if y is not None:												# labels available
				#note if useOutputConnectionsLastLayer, only the last index in hiddenNeuronPredictionAccuracy will be valid

				for lidx, act in enumerate(layerActivations):
					
					# 1) normalise & soft-max output weights
					w_norm  = self._normaliseOutputConnectionWeights(self.outputConnectionMatrix[lidx])
					w_soft  = torch.softmax(w_norm, dim=-1)						# same shape as outputConnectionMatrix

					active_mask = (act != 0)								# [B,H]

					soft_layer = w_soft
					conn_layer = (self.outputConnectionMatrix[lidx] != 0).any(dim=1)	# [H]

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


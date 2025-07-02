"""EISANIpt_EISANImodelSequentialDynamic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model Sequential Dynamic neuron segment connection assignment

"""

import torch
from ANNpt_globalDefs import *


# ----------------------------------------------------------------- #
# Forward propagation - slice to "used" portion only
# ----------------------------------------------------------------- #
def forwardProp(self, activationsLayer1: torch.Tensor, hiddenLayerIdx: int, pairId: int) -> torch.Tensor:
	indexArray  = (self.indexArrayA if pairId == 0 else self.indexArrayB)[hiddenLayerIdx]	# [capacity]

	# Gather - first replace "-1" placeholders by a safe 0 (any valid column)
	safeIndex = indexArray.clamp(min=0)
	acts = activationsLayer1.index_select(dim=1, index=safeIndex)				# [B, capacity]

	# Zero-out columns that correspond to unassigned slots
	mask = indexArray.unsqueeze(0).ne(-1)										# [1, capacity]
	activationsLayer2 = acts * mask.to(dtype=acts.dtype)
	return activationsLayer2

# ----------------------------------------------------------------- #
# Dynamic growth
# ----------------------------------------------------------------- #
def sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx: int, activationsLayerA1: torch.Tensor, activationsLayerB1: torch.Tensor) -> None:
	device = activationsLayerA1.device
	activeA = activationsLayerA1.any(dim=0)
	activeB = activationsLayerB1.any(dim=0)
	idxA = activeA.nonzero(as_tuple=False).flatten()
	idxB = activeB.nonzero(as_tuple=False).flatten()
	if idxA.numel() == 0 or idxB.numel() == 0:
		#print("idxA.numel() == 0 or idxB.numel() == 0")
		return

	candidatePairs = torch.cartesian_prod(idxA, idxB)				# [P, 2]
	uniqueMask  = perform_uniqueness_check(self, hiddenLayerIdx, candidatePairs)
	newPairs = candidatePairs[uniqueMask]
	nNew = newPairs.size(0)
	if nNew == 0:
		#print("nNew == 0")
		return

	# ensure capacity **for this layer only**
	if(debugSequentialSANIactivations):
		print("self.numAssignedNeuronSegments[hiddenLayerIdx] = ",  self.numAssignedNeuronSegments[hiddenLayerIdx])
	start = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
	end = start + nNew
	if end > self.indexArrayA[hiddenLayerIdx].size(0):
		expandArrays(self, hiddenLayerIdx, end - self.indexArrayA[hiddenLayerIdx].size(0))

	self.indexArrayA[hiddenLayerIdx][start:end] = newPairs[:, 0]
	self.indexArrayB[hiddenLayerIdx][start:end] = newPairs[:, 1]
	self.numAssignedNeuronSegments[hiddenLayerIdx] += nNew

# ----------------------------------------------------------------- #
# Uniqueness check
# ----------------------------------------------------------------- #
def perform_uniqueness_check(self, hiddenLayerIdx: int, candidatePairs: torch.Tensor) -> torch.Tensor:
	used = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
	if used == 0:
		return torch.ones(candidatePairs.size(0), dtype=torch.bool, device=candidatePairs.device)

	# ----  Hash each (A, B) pair into one 64-bit integer  --------------------
	# Safe for layer sizes < 2³². If you ever exceed that, switch to 128-bit
	# packing or use A * maxPrev + B instead.
	def _pack(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return (a.to(torch.int64) << 32) | b.to(torch.int64)

	existingKeys = _pack(self.indexArrayA[hiddenLayerIdx][:used], self.indexArrayB[hiddenLayerIdx][:used])			# [used]
	candidateKeys = _pack(candidatePairs[:, 0], candidatePairs[:, 1])						# [P]

	# torch.isin is O(N) and allocates only the output mask
	dup = torch.isin(candidateKeys, existingKeys)						# [P] bool
	return ~dup		

'''
#orig before optimisation

def perform_uniqueness_check(self, hiddenLayerIdx: int, candidatePairs: torch.Tensor) -> torch.Tensor:
	used = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
	if used == 0:
		return torch.ones(candidatePairs.size(0), dtype=torch.bool, device=candidatePairs.device)

	existA = self.indexArrayA[hiddenLayerIdx][:used].unsqueeze(0)	# [1, used]
	existB = self.indexArrayB[hiddenLayerIdx][:used].unsqueeze(0)	# [1, used]
	candA  = candidatePairs[:, 0].unsqueeze(1)				# [P, 1]
	candB  = candidatePairs[:, 1].unsqueeze(1)				# [P, 1]
	dup    = ((candA == existA) & (candB == existB)).any(dim=1)
	return ~dup	
'''

# ----------------------------------------------------------------- #
# Per-layer expansion (only A & B for the specified layer)
# ----------------------------------------------------------------- #
def expandArrays(self, hiddenLayerIdx: int, additionalRequired: int) -> None:
	#must always sync expansions with the arrays defined in EISANImodel:init();
	#print("expandArrays")

	if additionalRequired <= 0:
		return

	nBlocks = (additionalRequired + blockExpansionSize - 1) // blockExpansionSize
	growBy  = nBlocks * blockExpansionSize
	device  = self.indexArrayA[hiddenLayerIdx].device

	#expand connection arrays;
	indexPadding = torch.full((growBy,), -1, dtype=torch.long, device=device)
	self.indexArrayA[hiddenLayerIdx] = torch.cat([self.indexArrayA[hiddenLayerIdx], indexPadding], dim=0)
	self.indexArrayB[hiddenLayerIdx] = torch.cat([self.indexArrayB[hiddenLayerIdx], indexPadding.clone()], dim=0)

	#expand activation arrays (new sizes will be retained when processing next batch during reinitialisation);
	layerIdx = hiddenLayerIdx+1
	activationPadding = torch.zeros((self.batchSize, growBy,), dtype=torch.bool, device=device)
	self.layerActivation[layerIdx] = torch.cat([self.layerActivation[layerIdx], activationPadding.bool()], dim=1)
	self.layerActivationTime[layerIdx] = torch.cat([self.layerActivationTime[layerIdx], activationPadding.int()], dim=1)
	if(self.useSequentialSANIactivationStrength):
		self.layerActivationDistance[layerIdx] = torch.cat([self.layerActivationDistance[layerIdx], activationPadding.int()], dim=1)
		self.layerActivationCount[layerIdx] = torch.cat([self.layerActivationCount[layerIdx], activationPadding.int()], dim=1)
	self.layerActivationStrength = torch.cat([self.layerActivationStrength, activationPadding.float()], dim=1)	#temporary var

	#expand output arrays;
	if(not useOutputConnectionsLastLayer or hiddenLayerIdx==self.numberUniqueHiddenLayers-1):
		if(useSparseOutputMatrix):
			mat = self.outputConnectionMatrix[hiddenLayerIdx]
			crow = mat.crow_indices()						# length = oldRows + 1
			col  = mat.col_indices()
			val  = mat.values()
			last = crow[-1]									# nnz so far

			# append 'growBy' copies of the same pointer => empty new rows
			new_crow = torch.cat([crow, last.repeat(growBy)])
			new_rows = mat.size(0) + growBy
			new_size = (new_rows, mat.size(1))

			self.outputConnectionMatrix[hiddenLayerIdx] = torch.sparse_csr_tensor(new_crow, col, val, size=new_size, device=mat.device, dtype=mat.dtype)
		else:
			# 2.  Grow the dense output-weight matrix with zero rows
			outCols = self.outputConnectionMatrix[hiddenLayerIdx].size(1)		# keep column count unchanged
			outputConnectionPadding = torch.zeros((growBy, outCols), dtype=self.outputConnectionMatrix[hiddenLayerIdx].dtype, device=device)
			self.outputConnectionMatrix[hiddenLayerIdx] = torch.cat([self.outputConnectionMatrix[hiddenLayerIdx], outputConnectionPadding], dim=0)

	if(limitConnections):
		#expand accuracy tracker arrays;
		if(limitOutputConnections and limitOutputConnectionsBasedOnAccuracy):
			accPadding = torch.zeros((growBy, 2), dtype=torch.bool, device=device)
			self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx] = torch.cat([self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx], accPadding], dim=0)
		#expand usage arrays;
		usagePadding = torch.zeros((growBy,), dtype=torch.long, device=device)
		self.hiddenNeuronUsage[hiddenLayerIdx] = torch.cat([self.hiddenNeuronUsage[hiddenLayerIdx], usagePadding], dim=0)

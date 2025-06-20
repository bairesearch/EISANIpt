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

if(useConnectionWeights):

	def forwardProp(self, prevActivation: torchTensor, hiddenLayerIdx: int, segmentIdx: int, device: torch.device) -> torch.Tensor:	
		# prevActivation is torch.int8 (0 or 1)
		weight = self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIdx].to(device)

		dev	= prevActivation.device
		# weight = self.hiddenConnectionMatrix[hiddenLayerIdx].to(dev) # Already done above

		if useSparseHiddenMatrix:
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

		return z_float
			
	def sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx: int, prevActivationSeg0: torch.Tensor, prevActivationSeg1: torch.Tensor, device: torch.device, segIndex0: int = 0, segIndex1: int = 1,) -> None:
		"""
		Simplified growth rule: create a new hidden neuron for every combination
		of active presynaptic neurons (idx0 E prevActivationSeg0, idx1 E prevActivationSeg1).

		Each new neuron receives exactly one synapse on each of the two given
		segments:
			- segment segIndex0 -> idx0 (weight = +1)
			- segment segIndex1 -> idx1 (weight = +1)

		Growth is skipped if the (idx0, idx1) pair has been used before or if no
		free neurons remain that are un-assigned on *both* segments.
		"""

		# ------------------------------------------------------------------
		# 1. Identify currently active presynaptic indices for each segment
		# ------------------------------------------------------------------
		active0 = (prevActivationSeg0 > 0).any(dim=0).nonzero(as_tuple=True)[0]	# 1-D LongTensor
		active1 = (prevActivationSeg1 > 0).any(dim=0).nonzero(as_tuple=True)[0]

		if active0.numel() == 0 or active1.numel() == 0:
			return		# nothing to pair

		# ------------------------------------------------------------------
		# 2. Prepare bookkeeping structures (create once on first call)
		# ------------------------------------------------------------------
		pairSigDict = self.hiddenNeuronPairSignatures[hiddenLayerIdx]

		# ------------------------------------------------------------------
		# 3. List neurons still free on *both* segments
		# ------------------------------------------------------------------
		freeMask = (~self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex0] &
					~self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex1])
		freeList = freeMask.nonzero(as_tuple=True)[0]				# 1-D LongTensor

		if freeList.numel() == 0:
			if self.training:
				print(f"Warning: no free neurons left for pairwise growth in layer {hiddenLayerIdx}")
			return

		newIdxCursor = 0		# walk through freeList as we allocate

		# ------------------------------------------------------------------
		# 4. Generate a hidden neuron for every (idx0, idx1) pair
		# ------------------------------------------------------------------
		for idx0 in active0.tolist():
			for idx1 in active1.tolist():

				# --- 4a: uniqueness check -------------------------------------------------
				pairSig = f"{idx0},{idx1}"
				if pairSig in pairSigDict:
					continue		# pair already realised

				# --- 4b: ensure we still have capacity -----------------------------------
				if newIdxCursor >= freeList.numel():
					if self.training:
						print(f"Warning: ran out of neurons in layer {hiddenLayerIdx} while forming pairs")
					return

				newNeuronIdx = freeList[newIdxCursor].item()
				newIdxCursor += 1

				# --- 4c: mark the neuron as assigned on both segments --------------------
				self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex0, newNeuronIdx] = True
				self.neuronSegmentAssignedMask[hiddenLayerIdx, segIndex1, newNeuronIdx] = True

				# --- 4d: connect idx0 on segIndex0, idx1 on segIndex1 --------------------
				self._assign_single_connection(hiddenLayerIdx, segIndex0, newNeuronIdx, idx0, device)
				self._assign_single_connection(hiddenLayerIdx, segIndex1, newNeuronIdx, idx1, device)

				# --- 4e: remember the signature so we never duplicate this pair ----------
				pairSigDict[pairSig] = True


	# --------------------------------------------------------------------------
	# Helper: add exactly one synapse (weight = +1 | True) to a sparse or dense
	#		 connection matrix in-place, then write the updated matrix back.
	# --------------------------------------------------------------------------
	def _assign_single_connection(self, hiddenLayerIdx: int, segIdx: int, neuronIdx: int, prevIdx: int, device: torch.device,) -> None:

		mat = self.hiddenConnectionMatrix[hiddenLayerIdx][segIdx]

		if self.config.useSparseHiddenMatrix:
			existing_indices = mat._indices()
			existing_values  = mat._values()

			new_indices = torch.tensor([[neuronIdx], [prevIdx]], dtype=torch.long, device=device)
			new_values  = torch.ones(1, dtype=existing_values.dtype, device=device)

			mat = torch.sparse_coo_tensor(
				torch.cat([existing_indices, new_indices], dim=1),
				torch.cat([existing_values,  new_values ], dim=0),
				mat.size(),
				device=device,
			).coalesce()
		else:
			mat[neuronIdx, prevIdx] = 1.0

		self.hiddenConnectionMatrix[hiddenLayerIdx][segIdx] = mat.to(device)

else:

	# ----------------------------------------------------------------- #
	# Forward propagation - slice to "used" portion only
	# ----------------------------------------------------------------- #
	def forwardProp(self, activationsLayer1: torch.Tensor, hiddenLayerIdx: int, pairId: int) -> torch.Tensor:
		indexArray  = (self.indexArrayA if pairId == 0 else self.indexArrayB)[hiddenLayerIdx]	# [capacity]

		# Gather - first replace "-1" placeholders by a safe 0 (any valid column)
		safeIndex   = indexArray.clamp(min=0)
		acts        = activationsLayer1.index_select(dim=1, index=safeIndex)				# [B, capacity]

		# Zero-out columns that correspond to unassigned slots
		mask        = indexArray.unsqueeze(0).ne(-1)										# [1, capacity]
		activationsLayer2 = acts * mask.to(dtype=acts.dtype)
		return activationsLayer2

	# ----------------------------------------------------------------- #
	# Dynamic growth
	# ----------------------------------------------------------------- #
	def sequentialSANI_dynamic_hidden_growth_pairwise(self, hiddenLayerIdx: int, activationsLayerA1: torch.Tensor, activationsLayerB1: torch.Tensor) -> None:
		device = activationsLayerA1.device
		activeA = activationsLayerA1.any(dim=0)
		activeB = activationsLayerB1.any(dim=0)
		idxA    = activeA.nonzero(as_tuple=False).flatten()
		idxB    = activeB.nonzero(as_tuple=False).flatten()
		if idxA.numel() == 0 or idxB.numel() == 0:
			#print("idxA.numel() == 0 or idxB.numel() == 0")
			return

		candidatePairs = torch.cartesian_prod(idxA, idxB)				# [P, 2]
		uniqueMask     = perform_uniqueness_check(self, hiddenLayerIdx, candidatePairs)
		newPairs       = candidatePairs[uniqueMask]
		nNew           = newPairs.size(0)
		if nNew == 0:
			#print("nNew == 0")
			return

		# ensure capacity **for this layer only**
		if(debugSequentialSANIactivations):
			print("self.numAssignedNeuronSegments[hiddenLayerIdx] = ",  self.numAssignedNeuronSegments[hiddenLayerIdx])
		start = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
		end   = start + nNew
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
		# Safe for layer sizes < 2��. If you ever exceed that, switch to 128-bit
		# packing or use A * maxPrev + B instead.
		def _pack(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
			return (a.to(torch.int64) << 32) | b.to(torch.int64)

		existingKeys  = _pack(self.indexArrayA[hiddenLayerIdx][:used], self.indexArrayB[hiddenLayerIdx][:used])			# [used]
		candidateKeys = _pack(candidatePairs[:, 0], candidatePairs[:, 1])						# [P]

		# torch.isin is O(N) and allocates only the output mask
		dup = torch.isin(candidateKeys, existingKeys)						# [P] bool
		return ~dup		

	# ----------------------------------------------------------------- #
	# Per-layer expansion (only A & B for the specified layer)
	# ----------------------------------------------------------------- #
	def expandArrays(self, hiddenLayerIdx: int, additionalRequired: int) -> None:
		#must always sync expansions with the arrays defined in EISANImodel:init();
		
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
		activationPadding = torch.zeros((self.config.batchSize, growBy,), dtype=torch.bool, device=device)
		self.layerActivation[layerIdx] = torch.cat([self.layerActivation[layerIdx], activationPadding.bool()], dim=1)
		self.layerActivationTime[layerIdx] = torch.cat([self.layerActivationTime[layerIdx], activationPadding.int()], dim=1)
		if(useSequentialSANIactivationStrength):
			self.layerActivationDistance[layerIdx] = torch.cat([self.layerActivationDistance[layerIdx], activationPadding.int()], dim=1)
			self.layerActivationCount[layerIdx] = torch.cat([self.layerActivationCount[layerIdx], activationPadding.int()], dim=1)
			#self.layerActivationStrength[layerIdx] = torch.cat([self.layerActivationStrength[layerIdx], activationPadding.float()], dim=1)

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

		#expand accuracy tracker arrays;
		if(limitOutputConnectionsBasedOnAccuracy):
			accPadding = torch.zeros((growBy, 2), dtype=torch.bool, device=device)
			self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx] = torch.cat([self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx], accPadding], dim=0)



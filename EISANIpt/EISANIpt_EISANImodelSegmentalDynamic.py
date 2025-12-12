"""EISANIpt_EISANImodelSegmentalDynamic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model Segmental Dynamic neuron segment connection assignment

"""

import torch
from ANNpt_globalDefs import *


# ---------------------------------------------------------
# Neuron segment (connections) uniquesness checks;
# ---------------------------------------------------------

'''
# 64-bit modulus - a large prime < 26  (fits signed int64)
_MOD64 = 9223372036854775783
# base for polynomial rolling hash (also prime)
_BASE  = 1099511628211

def hash_connections(cols: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	"""
	Compute order-independent 64-bit hashes for a *batch* of neurons
	fully vectorised (no Python loop).

	Parameters
	----------
	cols, w : (G, k)  - presynaptic indices and weights (1)

	Returns
	-------
	hashes : (G,) int64  - per-neuron hash in [0, _MOD64-1]
	"""
	G, k = cols.shape
	device = cols.device

	# 1. sort rows so representation is order-invariant
	sort_idx   = torch.argsort(cols, dim=1)
	cols_sorted = torch.gather(cols, 1, sort_idx)
	w_sorted    = torch.gather(w,    1, sort_idx)

	# 2. encode (column, weight) - small int  (weight -1-0, +1-1)
	key_row = cols_sorted.to(torch.int64).mul_(2).add_((w_sorted > 0).to(torch.int64))  # (G,k)

	# 3. vectorised polynomial hash  S key[i] * BASE^{k-1-i}  (mod M)
	powers = torch.pow(_BASE, torch.arange(k - 1, -1, -1, device=device, dtype=torch.int64),) 	           .remainder(_MOD64)           # (k,)

	hashes = (key_row * powers).remainder(_MOD64).sum(dim=1).remainder(_MOD64)
	return hashes

# ------------------------------------------------------------------
# Single-row uniqueness check (now hash-based, no Python dicts)
# ------------------------------------------------------------------

def perform_uniqueness_check(
		self, hiddenLayerIdx: int,
		newNeuronIdx: int,            # scalar index within hidden layer
		cols: torch.Tensor,           # (k,)  presynaptic indices
		w: torch.Tensor               # (k,)  weights 1
		, segmentIndexToUpdate: int    # Added: segment index to update
) -> bool:
	"""
	Non-batch version for backward-compat.
	Returns
	-------
	unique : bool - True if neuron was not a duplicate.
	"""
	h = hash_connections(cols.unsqueeze(0), w.unsqueeze(0))[0]   # int64

	if useEIneurons:
		half = self.config.hiddenLayerSize // 2
		if newNeuronIdx < half:
			bank = self.hiddenHashesExc[hiddenLayerIdx][segmentIndexToUpdate] # Modified
		else:
			bank = self.hiddenHashesInh[hiddenLayerIdx][segmentIndexToUpdate] # Modified
	else:
		bank = self.hiddenHashes[hiddenLayerIdx][segmentIndexToUpdate] # Modified

	is_dup = torch.isin(h, bank)

	if is_dup:
		# abort growth for this neuron segment
		self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, newNeuronIdx] = False # Modified
		return False

	# record new hash
	if useEIneurons:
		if newNeuronIdx < half:
			self.hiddenHashesExc[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat([bank, h.unsqueeze(0)]) # Modified
		else:
			self.hiddenHashesInh[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat([bank, h.unsqueeze(0)]) # Modified
	else:
		self.hiddenHashes[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat([bank, h.unsqueeze(0)]) # Modified

	return True


def perform_uniqueness_check_vectorised(
		self, hiddenLayerIdx: int, colsBatch: torch.Tensor, wBatch: torch.Tensor,
		newRows: torch.Tensor,
		segmentIndexToUpdate: int # Added: segment index to update
):
	"""
	Vectorised duplicate elimination.
	colsBatch, wBatch : (G,k)
	newRows           : (G,) global row indices in hidden layer.

	Returns
	-------
	keep_mask : (G,) bool - True = keep neuron
	dup_found : bool      - True if any duplicate row detected
	"""
	hashes = hash_connections(colsBatch, wBatch)        # (G,)

	if useEIneurons:
		half = self.config.hiddenLayerSize // 2
		isExc = newRows < half

		# -------- excitatory ---------------
		exc_keep = ~torch.isin(hashes[isExc], self.hiddenHashesExc[hiddenLayerIdx][segmentIndexToUpdate]) # Modified
		if exc_keep.any():
			self.hiddenHashesExc[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat( # Modified
				[self.hiddenHashesExc[hiddenLayerIdx][segmentIndexToUpdate], hashes[isExc][exc_keep]])

		# -------- inhibitory ---------------
		inh_keep = ~torch.isin(hashes[~isExc], self.hiddenHashesInh[hiddenLayerIdx][segmentIndexToUpdate]) # Modified
		if inh_keep.any():
			self.hiddenHashesInh[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat( # Modified
				[self.hiddenHashesInh[hiddenLayerIdx][segmentIndexToUpdate], hashes[~isExc][inh_keep]])

		# stitch back together
		keep_mask = torch.empty_like(hashes, dtype=torch.bool)
		keep_mask[isExc] = exc_keep
		keep_mask[~isExc] = inh_keep
	else:
		keep_mask = ~torch.isin(hashes, self.hiddenHashes[hiddenLayerIdx][segmentIndexToUpdate]) # Modified
		if keep_mask.any():
			self.hiddenHashes[hiddenLayerIdx][segmentIndexToUpdate] = torch.cat( # Modified
				[self.hiddenHashes[hiddenLayerIdx][segmentIndexToUpdate], hashes[keep_mask]])

	# Abort growth for duplicate neuron segments by updating neuronSegmentAssignedMask
	if (~keep_mask).any():
		duplicate_rows = newRows[~keep_mask]
		self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, duplicate_rows] = False # Modified

	dup_found = (~keep_mask).any().item()
	return keep_mask, dup_found
'''

def build_signature(self, cols: torch.Tensor, w: torch.Tensor) -> str:
	# cols, w  are 1-D tensors length k  (on GPU)
	# move to CPU (tiny) and build sorted signature
	pairs = sorted(zip(cols.cpu().tolist(), w.cpu().tolist()))
	return ''.join(f'{c}{int(v):+d}' for c, v in pairs)

def build_signature_vectorised(self, colsBatch: torch.Tensor, wBatch: torch.Tensor) -> list[str]:
	sigs = []
	for cols, w in zip(colsBatch, wBatch):
		sigs.append(build_signature(self, cols, w))
	return sigs

def perform_uniqueness_check(self, hiddenLayerIdx, newNeuronIdx, randIdx, weights, segmentIndexToUpdate):
	unique = True
	cfg	= self.config
	
	sig_new = build_signature(self, randIdx, weights)

	if useEIneurons:
		if useInhibition:
			half = cfg.hiddenLayerSize // 2
			if newNeuronIdx < half:
				sigDict = self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIndexToUpdate]
			else:
				sigDict = self.hiddenNeuronSignaturesInh[hiddenLayerIdx][segmentIndexToUpdate]
		else:
			# All neurons are excitatory when inhibition is disabled
			sigDict = self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIndexToUpdate]
	else:
		sigDict = self.hiddenNeuronSignatures[hiddenLayerIdx][segmentIndexToUpdate]

	if sig_new in sigDict:
		# duplicate -> abort growth
		self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, newNeuronIdx] = False
		unique = False
	else:
		# record signature (will keep dict small and incremental)
		sigDict[sig_new] = True

	return unique

def perform_uniqueness_check_vectorised(self, hiddenLayerIdx, colIdx, weights, newRows, segmentIndexToUpdate):
	unique = True
	cfg	= self.config

	batchSigs = build_signature_vectorised(self, colIdx, weights)	 # len G
	dup_found = False
	keep_list = []
	
	if useEIneurons:
		if useInhibition:
			half = cfg.hiddenLayerSize // 2
			# keep_mask = [] # Original comment, assuming keep_list is used
			for r, sig in zip(newRows.tolist(), batchSigs):
				if r < half:
					sigDict = self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIndexToUpdate]
				else:
					sigDict = self.hiddenNeuronSignaturesInh[hiddenLayerIdx][segmentIndexToUpdate]
				if sig in sigDict:
					keep_list.append(False)
					dup_found = True
				else:
					keep_list.append(True)
					sigDict[sig] = True			# record & keep
		else:
			# All neurons are excitatory when inhibition is disabled
			sigDict = self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIndexToUpdate]
			for sig in batchSigs:
				if sig in sigDict:
					keep_list.append(False)
					dup_found = True
				else:
					keep_list.append(True)
					sigDict[sig] = True
	else:
		sigDict = self.hiddenNeuronSignatures[hiddenLayerIdx][segmentIndexToUpdate]
		for sig in batchSigs:
			if sig in sigDict:
				keep_list.append(False)
				dup_found = True
			else:
				keep_list.append(True)
				sigDict[sig] = True
				
	keep_mask = torch.tensor(keep_list, device=newRows.device, dtype=torch.bool) # Changed device

	# Abort growth for duplicate neurons by updating neuronSegmentAssignedMask
	if (~keep_mask).any():
		duplicate_rows = newRows[~keep_mask]
		self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, duplicate_rows] = False
	
	return keep_mask, dup_found


# ---------------------------------------------------------
# Dynamic hidden growth helper
# ---------------------------------------------------------

def dynamic_hidden_growth(self, hiddenLayerIdx: int, prevActivation: torch.Tensor, currentActivation: torch.Tensor, device: torch.device, segmentIndexToUpdate: int) -> None: # Added segmentIndexToUpdate
	batchActiveMask = currentActivation != 0.0
	fractionActive = batchActiveMask.float().mean().item()
	if(debugEISANIfractionActivated):
		printf("fractionActive = ", fractionActive)
	if fractionActive >= targetActivationSparsityFraction:
		return  # sparsity satisfied

	# Need to activate a new neuron (one per call)
	available = (~self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, :]).nonzero(as_tuple=True)[0] # Modified
	if(debugEISANIdynamicUsage):
		printf("neuronSegmentAssignedMask available.numel() = ", available.numel())
	if available.numel() == 0:
		if self.training:
			print(f"Warning: no more available neurons in hidden layer {hiddenLayerIdx} for segment {segmentIndexToUpdate}") # Modified
		return
	newNeuronIdx = available[0].item()
	self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, newNeuronIdx] = True # Modified


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

	if useEIneurons:
		# -------------------------------------------------------
		# implementation 1d: Sample synapses - 50% active, 50% inactive
		# For useEIneurons=True;
		# - the excitatory neuron segment connections should be defined based on 50% active presynaptic layer excitatory neurons, and 50% inactive presynaptic layer inhibitory neurons.
		# - the inhibitory neuron segment connections should be defined based on 50% inactive presynaptic layer excitatory neurons, and 50% active presynaptic layer inhibitory neurons.
		# -------------------------------------------------------

		if useInhibition:
			# ---------- EI-specific pools ----------
			halfPrev = presyn.numel() // 2
			isExcPresyn = torch.arange(presyn.numel(), device=device) < halfPrev
			isInhPresyn = ~isExcPresyn

			excActiveIdx = (isExcPresyn & (presyn > 0)).nonzero(as_tuple=True)[0]
			excInactiveIdx = (isExcPresyn & (presyn == 0)).nonzero(as_tuple=True)[0]
			inhActiveIdx = (isInhPresyn & (presyn > 0)).nonzero(as_tuple=True)[0]
			inhInactiveIdx = (isInhPresyn & (presyn == 0)).nonzero(as_tuple=True)[0]

			halfThis = self.config.hiddenLayerSize // 2
			isExcNeur = newNeuronIdx < halfThis

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
		else:
			# When inhibition is disabled, all neurons are excitatory - sample from all available presynaptic neurons
			activeIdx = (presyn > 0).nonzero(as_tuple=True)[0]
			allIdx = torch.arange(presyn.numel(), device=device)
			
			# Sample all synapses from available neurons (bias towards active if useActiveBias)
			if useActiveBias and activeIdx.numel() > 0:
				# Prefer active neurons
				if activeIdx.numel() >= numSyn:
					choose = torch.randperm(activeIdx.numel(), device=device)[:numSyn]
					randIdx = activeIdx[choose]
				else:
					# Take all active, fill remainder from all neurons
					need = numSyn - activeIdx.numel()
					fill_idx = torch.randperm(allIdx.numel(), device=device)[:need]
					randIdx = torch.cat([activeIdx, allIdx[fill_idx]], dim=0)
					randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]
			else:
				# Random sampling from all neurons
				choose = torch.randperm(allIdx.numel(), device=device)[:numSyn]
				randIdx = allIdx[choose]

		# EI mode -> all weights are +1
		#weights = torch.ones_like(randIdx, dtype=torch.float32, device=device)	#orig
		if useSparseHiddenMatrix: #assuming E/I matrices have same dtype
			weights = torch.ones_like(randIdx, dtype=torch.bool, device=device)
		else:
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
		if useInhibition:
			if useActiveBias:
				numActive = (numSyn + 1) // 2		# ceiling - bias positive	# bias for odd k
			else:
				numActive = numSyn // 2	#orig
		else:
			numActive = numSyn
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

		chooseA = torch.randperm(activeIdx.numel(),   device=device)[:numActive] if numActive > 0 else torch.empty(0, dtype=torch.long, device=device)
		chooseI = torch.randperm(inactiveIdx.numel(), device=device)[:numInactive] if numInactive > 0 else torch.empty(0, dtype=torch.long, device=device)

		# Concatenate only non-empty tensors
		randIdx_parts = []
		if chooseA.numel() > 0:
			randIdx_parts.append(activeIdx[chooseA])
		if chooseI.numel() > 0:
			randIdx_parts.append(inactiveIdx[chooseI])
		
		if randIdx_parts:
			randIdx = torch.cat(randIdx_parts, dim=0)
		else:
			# Fallback to random selection if both pools are empty
			randIdx = torch.randperm(presyn.numel(), device=device)[:numSyn]

		# make sure we have exactly `numSyn` indices
		shortfall = numSyn - randIdx.numel()
		if shortfall > 0:
			# borrow from all available indices
			allIdx = torch.arange(presyn.numel(), device=device)
			extra = allIdx[torch.randperm(allIdx.numel(), device=device)[:shortfall]]
			randIdx = torch.cat([randIdx, extra], dim=0)
		randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]	#shuffle - permute so the order is still random
			
		# weights: +1 if presynaptic *currently on*, else -1
		prevActSample = presyn[randIdx]						# 0./1.
		#weights = torch.where(prevActSample > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))	#orig
		if useSparseHiddenMatrix:
			weights = presyn[randIdx] > 0	#boolean
		else:
			weights = torch.where(presyn[randIdx] > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))


	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		if not perform_uniqueness_check(self, hiddenLayerIdx, newNeuronIdx, randIdx, weights, segmentIndexToUpdate): # Added segmentIndexToUpdate
			#if(debugEISANIdynamicUsage):
			#	printf("dynamic_hidden_growth warning: generated neuron segment not unique")
			return
			
	# Update hidden connection matrix
	if useEIneurons:
		# Decide whether the new neuron is excitatory or inhibitory based on index
		half = self.config.hiddenLayerSize // 2
		if newNeuronIdx < half or not useInhibition:
			mat = self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIndexToUpdate]
			relativeIdx = newNeuronIdx if not useInhibition else newNeuronIdx
		else:
			mat = self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIndexToUpdate]
			relativeIdx = newNeuronIdx - half
	else:
		mat = self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIndexToUpdate]
		relativeIdx = newNeuronIdx

	if useSparseHiddenMatrix:
		# Append to sparse matrix (now 3D)
		existing_indices = mat._indices() # [2, nnz_old]
		existing_values = mat._values()   # [nnz_old]

		# New entries for the specific segment
		num_new_synapses = randIdx.numel()
		new_neuron_indices = torch.full((num_new_synapses,), relativeIdx, device=device, dtype=torch.long)
		new_prev_neuron_indices = randIdx

		new_indices = torch.stack([new_neuron_indices, new_prev_neuron_indices], dim=0) # [2, num_new_synapses] # Modified
		new_values = weights #dtype is bool or float32

		dev = existing_indices.device
		new_indices = new_indices.to(dev)
		new_values  = new_values.to(dev)

		matNew = torch.sparse_coo_tensor(torch.cat([existing_indices, new_indices], dim=1), torch.cat([existing_values, new_values]), mat.size(), device=dev,).coalesce()
	else:
		mat[relativeIdx, randIdx] = weights
		matNew = mat

	if useEIneurons and newNeuronIdx >= half and useInhibition:
		self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIndexToUpdate] = matNew.to(device) #ensure device consistency
	elif useEIneurons:
		self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIndexToUpdate] = matNew.to(device) #ensure device consistency
	else:
		self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIndexToUpdate] = matNew.to(device) #ensure device consistency

def dynamic_hidden_growth_vectorised(self, hiddenLayerIdx: int, prevActivation: torch.Tensor, currActivation: torch.Tensor, device: torch.device, segmentIndexToUpdate: int) -> None: # Added segmentIndexToUpdate
	"""
	Vectorised growth: handles an entire batch, but creates at most ONE new
	neuron per *sample* that is below the sparsity target.  Exactly follows
	the spec for both EI and non-EI modes.

	prevActivation/currActivation: [B, prevSize] float {0,1}
	"""
	cfg = self.config
	k = cfg.numberOfSynapsesPerSegment
	B, P = prevActivation.shape
	Lsize = currActivation.size(1)

	# ---------- 1. which samples need a neuron? ------------------------------
	fracAct = currActivation.float().mean(dim=1)				 # [B]
	if(debugEISANIfractionActivated):
		#print("prevActivation.shape = ", prevActivation.shape)
		#print("prevActivation.count_nonzero(dim=0).count_nonzero(dim=0) = ", prevActivation.count_nonzero(dim=0).count_nonzero(dim=0))
		#print("currActivation.shape = ", currActivation.shape)
		#print("currActivation.count_nonzero(dim=1).count_nonzero(dim=0) = ", currActivation.count_nonzero(dim=1).count_nonzero(dim=0))
		print("fracAct = ", fracAct, ", total = ", currActivation.shape[1])
	growMask = fracAct < targetActivationSparsityFraction   # bool [B]
	if not growMask.any():
		return

	growSamples = growMask.nonzero(as_tuple=False).squeeze(1)	# [G] sample idx
	G = growSamples.numel()

	# ---------- 2. reserve G unused neuron slots for the specified segment -----------------------------
	avail = (~self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, :]).nonzero(as_tuple=True)[0] # Modified
	if(debugEISANIdynamicUsage):
		printf("neuronSegmentAssignedMask avail.numel() = ", avail.numel())
	if avail.numel() < G:
		G = avail.numel()
		growSamples = growSamples[:G]
		if G == 0:
			if self.training:
				print(f"Warning: no free neurons in layer {hiddenLayerIdx} for segment {segmentIndexToUpdate}") # Modified
			return
	newRows = avail[:G] # [G] neuron indices
	self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIndexToUpdate, newRows] = True # Modified
	#print("G grow samples = ", G)

	# ---------- 3. vectorised synapse sampling -------------------------------
	# presynBatch: [G, P] float {0,1}
	presynBatch = prevActivation[growSamples]

	onMask	  = presynBatch > 0								# bool
	offMask	 = ~onMask

	# decide counts
	if useInhibition:
		if useActiveBias:
			nA = (k + 1) // 2	# ceil(k/2)
		else:
			nA = k // 2
	else:
		nA = k
	nI = k - nA

	# EI-aware pools -----------------------------------------------------------
	if useEIneurons:
		halfPrev = P // 2
		isExcPre = torch.arange(P, device=device) < halfPrev   # [P] bool
		isInhPre = ~isExcPre

		# split masks
		excOn = onMask & isExcPre
		excOff = offMask & isExcPre
		inhOn = onMask & isInhPre
		inhOff = offMask & isInhPre

		# neuron type for each new row
		halfThis  = cfg.hiddenLayerSize // 2
		if useInhibition:
			isExcNeur = newRows < halfThis						   # [G] bool
		else:
			isExcNeur = torch.ones_like(newRows, dtype=torch.bool)   # All neurons are excitatory when inhibition is disabled

		# choose pools per neuron in vectorised form
		activePool   = torch.where(isExcNeur.unsqueeze(1), excOn,  inhOn if useInhibition else excOn)
		inactivePool = torch.where(isExcNeur.unsqueeze(1), inhOff, excOff if useInhibition else excOff)

		actIdx = draw_indices(activePool,   nA) if nA > 0 else torch.empty(G, 0, dtype=torch.long, device=device)				 # [G, nA]
		inIdx = draw_indices(inactivePool, nI) if nI > 0 else torch.empty(G, 0, dtype=torch.long, device=device)				 # [G, nI]
		colIdx = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]
		#weights = torch.ones(G, k, device=device)				# all +1	#orig
		if useSparseHiddenMatrix: #assuming E/I matrices have same dtype
			weights = torch.ones(G, k, device=device, dtype=torch.bool)
		else:
			weights = torch.ones(G, k, device=device, dtype=torch.float32)
	else:
		# non-EI: 50 / 50 active / inactive
		actIdx = draw_indices(onMask,  nA) if nA > 0 else torch.empty(G, 0, dtype=torch.long, device=device)					  # [G, nA]
		inIdx = draw_indices(offMask, nI) if nI > 0 else torch.empty(G, 0, dtype=torch.long, device=device)					  # [G, nI]
		colIdx = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]

		presynPicked = presynBatch.gather(1, colIdx)			 # 0./1., [G,k]
		#weights = torch.where(presynPicked > 0, torch.ones_like(presynPicked), -torch.ones_like(presynPicked))	# 1	#orig
		if useSparseHiddenMatrix:
			weights = presynPicked > 0 #boolean
		else:
			weights = torch.where(presynPicked > 0, torch.ones_like(presynPicked), -torch.ones_like(presynPicked))
	
	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		keep_mask, dup_found = perform_uniqueness_check_vectorised(self, hiddenLayerIdx, colIdx, weights, newRows, segmentIndexToUpdate) # Added segmentIndexToUpdate
		if keep_mask.any():
			#filter out rows whose signature already exists
			newRows = newRows[keep_mask]
			colIdx = colIdx[keep_mask]
			weights = weights[keep_mask]
			G = newRows.numel()
			#if(debugEISANIdynamicUsage):
			#	if dup_found:
			#		printf("dynamic_hidden_growth_vectorised warning: non-unique generated neuron segments (ie duplicates) skipped")
		else:
			#if(debugEISANIdynamicUsage):
			#	printf("dynamic_hidden_growth_vectorised warning: no generated neuron segments unique in this batch")
			return

	# shuffle cols inside each row to keep random order
	perm = torch.randperm(k, device=device)
	colIdx = colIdx[:, perm]
	weights = weights[:, perm]

	# ---------- 4. flatten to COO lists for 3D sparse tensor --------------------------------------
	num_new_connections_total = G * k # Corrected G if it was reduced by uniqueness check
	flatNeuronIndices = newRows.repeat_interleave(k) # [G*k] - these are the neuron indices (0 to numNeurons-1)
	flatPrevNeuronIndices = colIdx.reshape(-1) # [G*k] - these are the presynaptic neuron indices
	flatVals = weights.reshape(-1) # [G*k]

	if useEIneurons:
		# ---------- 5. write to weight matrix (sparse or dense) ------------------
		half = cfg.hiddenLayerSize // 2
		# excMask = newRows < half						# [G] bool # Not needed here, use flatNeuronIndices
		# inhMask = ~excMask

		def merge_into(mat, rowsSel, colsSel, valsSel, isExc):
			"""
			rowsSel : flat (global) neuron indices
			colsSel : flat presynapse indices
			valsSel : flat weights
			isExc   : True -> excitatory matrix, row shift = 0
					  False -> inhibitory  matrix, row shift = -half
			"""
			if rowsSel.numel() == 0:
				return mat
			rowShift = 0 if isExc else -half
			rowsRel  = rowsSel + rowShift # these are neuron indices relative to E or I part

			if useSparseHiddenMatrix:
				# For 3D sparse: indices are [neuron_idx, segment_idx, prev_neuron_idx]
				num_entries = rowsRel.numel()
				idx_new = torch.stack([rowsRel, colsSel], dim=0) # [2, num_entries] # Modified
				val_new = valsSel #dtype is bool or float32
				mat = torch.sparse_coo_tensor(torch.cat([mat.indices(), idx_new], dim=1), torch.cat([mat.values(),  val_new]), size=mat.size(), device=device,).coalesce()
			else: # dense 3D matrix
				mat = mat.clone()
				# Ensure valsSel is compatible with int8 matrix
				if mat.dtype == torch.int8:
					# Assuming valsSel from sparse bool context is True/False, convert to 1/0 or 1/-1 for int8
					if valsSel.dtype == torch.bool:
						# Standard case: True -> 1, False -> -1 (if not EI)
						# EI case: True -> 1 (False should not occur for EI weights)
						# This logic should align with how weights are determined for int8 dense elsewhere
						# For simplicity, assuming EI means valsSel are effectively 1 (True)
						# and non-EI means valsSel could be True (1) or False (-1)
						if useEIneurons: # context of merge_into being called
							mat[rowsRel, colsSel] = valsSel.to(torch.int8) # True -> 1
						else:
							mat[rowsRel, colsSel] = torch.where(valsSel, torch.tensor(1, device=device, dtype=torch.int8), torch.tensor(-1, device=device, dtype=torch.int8))
					else:
						mat[rowsRel, colsSel] = valsSel.to(torch.int8) # direct cast if already numeric # Modified
				else:
					mat[rowsRel, colsSel] = valsSel # Modified
			return mat

		# split flattened arrays once for efficiency
		isExcEntry = flatNeuronIndices < half # Mask for entries belonging to excitatory neurons
		
		# Excitatory neuron updates
		flatNeuronIndices_exc = flatNeuronIndices[isExcEntry]
		flatPrevNeuronIndices_exc = flatPrevNeuronIndices[isExcEntry]
		flatVals_exc = flatVals[isExcEntry]

		# Inhibitory neuron updates (only if inhibition is enabled)
		if useInhibition:
			flatNeuronIndices_inh = flatNeuronIndices[~isExcEntry]
			flatPrevNeuronIndices_inh = flatPrevNeuronIndices[~isExcEntry]
			flatVals_inh = flatVals[~isExcEntry]
		else:
			# When inhibition is disabled, all entries are excitatory
			if not isExcEntry.all():
				# Add any remaining entries to excitatory
				flatNeuronIndices_exc = flatNeuronIndices
				flatPrevNeuronIndices_exc = flatPrevNeuronIndices
				flatVals_exc = flatVals
			flatNeuronIndices_inh = torch.empty(0, dtype=flatNeuronIndices.dtype, device=device)
			flatPrevNeuronIndices_inh = torch.empty(0, dtype=flatPrevNeuronIndices.dtype, device=device)
			flatVals_inh = torch.empty(0, dtype=flatVals.dtype, device=device)

		# merge into respective matrices
		Emat = self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIndexToUpdate]
		Emat = merge_into(Emat, flatNeuronIndices_exc, flatPrevNeuronIndices_exc, flatVals_exc, True) # Added segmentIndexToUpdate
		self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIndexToUpdate] = Emat

		if useInhibition:
			Imat = self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIndexToUpdate]
			Imat = merge_into(Imat, flatNeuronIndices_inh, flatPrevNeuronIndices_inh, flatVals_inh, False) # Added segmentIndexToUpdate
			self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIndexToUpdate] = Imat
	else:
		# ---------- 5. write to weight matrix (sparse or dense) for standard (non-EI) ------------------
		mat = self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIndexToUpdate]

		if useSparseHiddenMatrix:
			# For 3D sparse: indices are [neuron_idx, segment_idx, prev_neuron_idx]
			idx_new = torch.stack([flatNeuronIndices, flatPrevNeuronIndices], dim=0) # [2, G*k] # Modified
			val_new = flatVals #dtype is bool or float32
			mat_new = torch.sparse_coo_tensor(torch.cat([mat.indices(), idx_new], dim=1), torch.cat([mat.values(),  val_new]), size=mat.size(), device=device,).coalesce()
		else:	# dense case (3D)
			mat_new = mat.clone()
			# Ensure flatVals is compatible with int8 matrix
			if mat_new.dtype == torch.int8:
				# flatVals can be bool (from sparse logic) or float (potentially, though aiming for int8)
				# This part handles the case where dense matrices are int8
				if flatVals.dtype == torch.bool:
					# Convert boolean flatVals to int8: True to 1, False to -1 (standard) or True to 1 (EI)
					# This context is non-EI, so True -> 1, False -> -1
					mat_new[flatNeuronIndices, flatPrevNeuronIndices] = torch.where(flatVals, torch.tensor(1, device=device, dtype=torch.int8), torch.tensor(-1, device=device, dtype=torch.int8)) 
				else:
					mat_new[flatNeuronIndices, flatPrevNeuronIndices] = flatVals.to(torch.int8) # direct cast if already numeric # Modified
			else:
				mat_new[flatNeuronIndices, flatPrevNeuronIndices] = flatVals # Modified

		# ---------- 6. store back -------------------------------------------------
		self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIndexToUpdate] = mat_new.to(device) #ensure device consistency


def draw_indices(onMask: torch.Tensor, n: int) -> torch.Tensor:
	"""
	Select `n` column indices per group where `onMask` is True.
	\u25ce Works even when C > 2**24 (torch.multinomial limit).

	Args
	----
	onMask : (G, C) bool   \u2013 mask of eligible presynaptic neurons.
	n      : int           \u2013 number of indices to draw per group (with
	                         replacement, uniform over True entries).

	Returns
	-------
	idx : (G, n) int64     \u2013 sampled column indices.
	"""
	G, C = onMask.shape

	# Handle case when n=0
	if n == 0:
		return torch.empty(G, 0, dtype=torch.long, device=onMask.device)

	if C <= (1 << 24):
		# normal path \u2013 multinomial is safe
		prob = onMask.float() + 1e-6							   # avoid zero row
		idx  = torch.multinomial(prob, n, replacement=True)	  # [G, n]
		return idx

	# ---------- big-width fallback ----------
	# Strategy: 1) get True positions per row, 2) sample with randint.
	# This keeps memory modest and is CUDA-friendly.

	# 1. list of index tensors, one per group
	true_per_row = onMask.nonzero(as_tuple=False)          # (K, 2)
	#   row_offsets[i] ... row_offsets[i+1]-1 are entries of group i
	row_offsets = torch.zeros(G + 1, dtype=torch.long, device=onMask.device)
	row_offsets[1:].copy_(torch.bincount(true_per_row[:, 0], minlength=G))
	row_offsets = row_offsets.cumsum(0)                    # (G+1,)

	# 2. sample
	samples = []
	for g in range(G):
		start, end = row_offsets[g].item(), row_offsets[g + 1].item()
		count      = end - start
		if count == 0:
			#v1;
			#samples.append(torch.full((n,), -1, device=onMask.device))  # no candidates
			
			#v2;
			# row has no eligible columns \u2013 pick any valid index (e.g. 0)
			#samples.append(torch.zeros(n, dtype=torch.long, device=onMask.device))
			
			#v3;
			samples.append(torch.randint(0, C, (n,), device=onMask.device))
			continue
		pool = true_per_row[start:end, 1]                   # (count,)
		rnd  = torch.randint(0, count, (n,), device=onMask.device)
		samples.append(pool[rnd])

	return torch.stack(samples)                            # (G, n)

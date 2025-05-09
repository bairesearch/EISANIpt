"""EISANIpt_EISANImodelDynamic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model dynamic neuron segment connection assignment

"""

import torch
from ANNpt_globalDefs import *
import EISANIpt_EISANImodelDynamic


# ---------------------------------------------------------
# Neuron segment (connections) uniquesness checks;
# ---------------------------------------------------------

def _build_signature(self, cols: torch.Tensor, w: torch.Tensor) -> str:
	# cols, w  are 1-D tensors length k  (on GPU)
	# move to CPU (tiny) and build sorted signature
	pairs = sorted(zip(cols.cpu().tolist(), w.cpu().tolist()))
	return ''.join(f'{c}{int(v):+d}' for c, v in pairs)

def _build_signature_vectorised(self, colsBatch: torch.Tensor, wBatch: torch.Tensor) -> list[str]:
	sigs = []
	for cols, w in zip(colsBatch, wBatch):
		sigs.append(_build_signature(self, cols, w))
	return sigs

def perform_uniqueness_check(self, layerIdx, newNeuronIdx, randIdx, weights):
	unique = True
	cfg	= self.config
	
	sig_new = _build_signature(self, randIdx, weights)

	if self.useEIneurons:
		half = cfg.hiddenLayerSize // 2
		if newNeuronIdx < half:
			sigDict = self.hiddenNeuronSignaturesExc[layerIdx]
		else:
			sigDict = self.hiddenNeuronSignaturesInh[layerIdx]
	else:
		sigDict = self.hiddenNeuronSignatures[layerIdx]

	if sig_new in sigDict:
		# duplicate -> abort growth
		self.neuronSegmentAssignedMask[layerIdx, newNeuronIdx] = False
		unique = False
	else:
		# record signature (will keep dict small and incremental)
		sigDict[sig_new] = True

	return unique

def perform_uniqueness_check_vectorised(self, layerIdx, colIdx, weights, newRows):
	unique = True
	cfg	= self.config

	batchSigs = _build_signature_vectorised(self, colIdx, weights)	 # len G
	dup_found = False
	keep_list = []
	
	if self.useEIneurons:
		half = cfg.hiddenLayerSize // 2
		# keep_mask = [] # Original comment, assuming keep_list is used
		for r, sig in zip(newRows.tolist(), batchSigs):
			if r < half:
				sigDict = self.hiddenNeuronSignaturesExc[layerIdx]
			else:
				sigDict = self.hiddenNeuronSignaturesInh[layerIdx]
			if sig in sigDict:
				keep_list.append(False)
				dup_found = True
			else:
				keep_list.append(True)
				sigDict[sig] = True			# record & keep
	else:
		sigDict = self.hiddenNeuronSignatures[layerIdx]
		for sig in batchSigs:
			if sig in sigDict:
				keep_list.append(False)
				dup_found = True
			else:
				keep_list.append(True)
				sigDict[sig] = True
				
	keep_mask = torch.tensor(keep_list, device=newRows.device, dtype=torch.bool) # Changed device
	return keep_mask, dup_found

# ---------------------------------------------------------
# Dynamic hidden growth helper
# ---------------------------------------------------------

@torch.no_grad()
def _dynamic_hidden_growth(self, layerIdx: int, prevActivation: torch.Tensor, currentActivation: torch.Tensor, device: torch.device,) -> None:
	batchActiveMask = currentActivation != 0.0
	fractionActive = batchActiveMask.float().mean().item()
	#if(debugEISANIoutput):
	#	print("fractionActive = ", fractionActive)
	if fractionActive >= self.targetActivationSparsityFraction:
		return  # sparsity satisfied

	# Need to activate a new neuron (one per call)
	available = (~self.neuronSegmentAssignedMask[layerIdx]).nonzero(as_tuple=True)[0]
	if(debugEISANIoutput):
		print("neuronSegmentAssignedMask available.numel() = ", available.numel())
	if available.numel() == 0:
		if self.training:
			print(f"Warning: no more available neurons in hidden layer {layerIdx}")
		return
	newNeuronIdx = available[0].item()
	self.neuronSegmentAssignedMask[layerIdx, newNeuronIdx] = True


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

	if self.useEIneurons:
		# -------------------------------------------------------
		# implementation 1d: Sample synapses - 50% active, 50% inactive
		# For useEIneurons=True;
		# - the excitatory neuron segment connections should be defined based on 50% active presynaptic layer excitatory neurons, and 50% inactive presynaptic layer inhibitory neurons.
		# - the inhibitory neuron segment connections should be defined based on 50% inactive presynaptic layer excitatory neurons, and 50% active presynaptic layer inhibitory neurons.
		# -------------------------------------------------------

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

		# EI mode -> all weights are +1
		#weights = torch.ones_like(randIdx, dtype=torch.float32, device=device)	#orig
		if self.hiddenConnectionMatrixExcitatory[layerIdx].is_sparse: #assuming E/I matrices have same dtype
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
		if useActiveBias:
			numActive = (numSyn + 1) // 2		# ceiling - bias positive	# bias for odd k
		else:
			numActive = numSyn // 2	#orig
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

		chooseA = torch.randperm(activeIdx.numel(),   device=device)[:numActive]
		chooseI = torch.randperm(inactiveIdx.numel(), device=device)[:numInactive]

		randIdx = torch.cat([activeIdx[chooseA], inactiveIdx[chooseI]], dim=0)
		# make sure we have exactly `numSyn` indices
		shortfall = numSyn - randIdx.numel()
		if shortfall > 0:
			# borrow from the other pool (wraps around if needed)
			filler = inactivePool if activePool.numel() >= inactivePool.numel() else activePool
			extra  = filler[torch.randperm(filler.numel(), device=device)[:shortfall]]
			randIdx = torch.cat([randIdx, extra], dim=0)
		randIdx = randIdx[torch.randperm(randIdx.numel(), device=device)]	#shuffle - permute so the order is still random
			
		# weights: +1 if presynaptic *currently on*, else -1
		prevActSample = presyn[randIdx]						# 0./1.
		#weights = torch.where(prevActSample > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))	#orig
		if self.hiddenConnectionMatrix[layerIdx].is_sparse:
			weights = presyn[randIdx] > 0	#boolean
		else:
			weights = torch.where(presyn[randIdx] > 0, torch.ones_like(prevActSample), -torch.ones_like(prevActSample))


	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		if not perform_uniqueness_check(self, layerIdx, newNeuronIdx, randIdx, weights):
			if(debugEISANIoutput):
				print("_dynamic_hidden_growth warning: generated neuron segment not unique")
			return
			
	# Update hidden connection matrix
	if self.useEIneurons:
		# Decide whether the new neuron is excitatory or inhibitory based on index
		half = self.config.hiddenLayerSize // 2
		if newNeuronIdx < half:
			mat = self.hiddenConnectionMatrixExcitatory[layerIdx]
			relativeIdx = newNeuronIdx
		else:
			mat = self.hiddenConnectionMatrixInhibitory[layerIdx]
			relativeIdx = newNeuronIdx - half
	else:
		mat = self.hiddenConnectionMatrix[layerIdx]
		relativeIdx = newNeuronIdx

	if mat.is_sparse:
		# Append to sparse matrix
		existing_indices = mat._indices()
		existing_values = mat._values()

		new_indices = torch.stack([torch.full((randIdx.numel(),), relativeIdx, device=device, dtype=torch.long), randIdx,])
		new_values = weights #dtype is bool or float32

		dev = existing_indices.device
		new_indices = new_indices.to(dev)
		new_values  = new_values.to(dev)

		matNew = torch.sparse_coo_tensor(torch.cat([existing_indices, new_indices], dim=1), torch.cat([existing_values, new_values]), mat.size(), device=dev,)
	else:
		# structural update - do NOT track in autograd
		with torch.no_grad():
			mat[relativeIdx, randIdx] = weights
		matNew = mat

	if self.useEIneurons and newNeuronIdx >= half:
		self.hiddenConnectionMatrixInhibitory[layerIdx] = matNew.to(device) #ensure device consistency
	elif self.useEIneurons:
		self.hiddenConnectionMatrixExcitatory[layerIdx] = matNew.to(device) #ensure device consistency
	else:
		self.hiddenConnectionMatrix[layerIdx] = matNew.to(device) #ensure device consistency

@torch.no_grad()
def _dynamic_hidden_growth_vectorised(self, layerIdx: int, prevActivation: torch.Tensor, currActivation: torch.Tensor, device: torch.device,) -> None:
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
	#if(debugEISANIoutput):
	#	print("fracAct = ", fracAct)
	growMask = fracAct < self.targetActivationSparsityFraction   # bool [B]
	if not growMask.any():
		return

	growSamples = growMask.nonzero(as_tuple=False).squeeze(1)	# [G] sample idx
	G = growSamples.numel()

	# ---------- 2. reserve G unused neuron slots -----------------------------
	avail = (~self.neuronSegmentAssignedMask[layerIdx]).nonzero(as_tuple=True)[0]
	if(debugEISANIoutput):
		print("neuronSegmentAssignedMask avail.numel() = ", avail.numel())
	if avail.numel() < G:
		G = avail.numel()
		growSamples = growSamples[:G]
		if G == 0:
			if self.training:
				print(f"Warning: no free neurons in layer {layerIdx}")
			return
	newRows = avail[:G]										  # [G]
	self.neuronSegmentAssignedMask[layerIdx, newRows] = True

	# ---------- 3. vectorised synapse sampling -------------------------------
	# presynBatch: [G, P] float {0,1}
	presynBatch = prevActivation[growSamples]
	onMask	  = presynBatch > 0								# bool
	offMask	 = ~onMask

	# -- helper to draw N unique indices from a row mask ----------------------
	def draw_indices(mask: torch.Tensor, n: int) -> torch.Tensor:
		"""
		mask : [G, P]  bool
		returns [G, n] long, allowing repeats if a row has < n True
		"""
		prob = mask.float() + 1e-6							   # avoid zero row
		idx  = torch.multinomial(prob, n, replacement=True)	  # [G, n]
		return idx

	# decide counts
	if getattr(self, "useActiveBias", True):
		nA = (k + 1) // 2	# ceil(k/2)
	else:
		nA = k // 2
	nI = k - nA

	# EI-aware pools -----------------------------------------------------------
	if self.useEIneurons:
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
		isExcNeur = newRows < halfThis						   # [G] bool

		# choose pools per neuron in vectorised form
		activePool   = torch.where(isExcNeur.unsqueeze(1), excOn,  inhOn)
		inactivePool = torch.where(isExcNeur.unsqueeze(1), inhOff, excOff)

		actIdx = draw_indices(activePool,   nA)				 # [G, nA]
		inIdx = draw_indices(inactivePool, nI)				 # [G, nI]
		colIdx = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]
		#weights = torch.ones(G, k, device=device)				# all +1	#orig
		if self.hiddenConnectionMatrixExcitatory[layerIdx].is_sparse: #assuming E/I matrices have same dtype
			weights = torch.ones(G, k, device=device, dtype=torch.bool)
		else:
			weights = torch.ones(G, k, device=device, dtype=torch.float32)
	else:
		# non-EI: 50 / 50 active / inactive
		actIdx = draw_indices(onMask,  nA)					  # [G, nA]
		inIdx = draw_indices(offMask, nI)					  # [G, nI]
		colIdx = torch.cat([actIdx, inIdx], dim=1)			  # [G, k]

		presynPicked = presynBatch.gather(1, colIdx)			 # 0./1., [G,k]
		#weights = torch.where(presynPicked > 0, torch.ones_like(presynPicked), -torch.ones_like(presynPicked))	# 1	#orig
		if self.hiddenConnectionMatrix[layerIdx].is_sparse:
			weights = presynPicked > 0 #boolean
		else:
			weights = torch.where(presynPicked > 0, torch.ones_like(presynPicked), -torch.ones_like(presynPicked))
	
	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		keep_mask, dup_found = perform_uniqueness_check_vectorised(self, layerIdx, colIdx, weights, newRows)
		if keep_mask.any():
			#filter out rows whose signature already exists
			newRows = newRows[keep_mask]
			colIdx = colIdx[keep_mask]
			weights = weights[keep_mask]
			G = newRows.numel()
			if(debugEISANIoutput):
				if dup_found:
					print("_dynamic_hidden_growth_vectorised warning: non-unique generated neuron segments (ie duplicates) skipped")
		else:
			if(debugEISANIoutput):
				print("_dynamic_hidden_growth_vectorised warning: no generated neuron segments unique in this batch")
			return

	# shuffle cols inside each row to keep random order
	perm = torch.randperm(k, device=device)
	colIdx = colIdx[:, perm]
	weights = weights[:, perm]

	# ---------- 4. flatten to COO lists --------------------------------------
	flatRows = newRows.repeat_interleave(k)					  # [G*k]
	flatCols = colIdx.reshape(-1)								# [G*k]
	flatVals = weights.reshape(-1)							   # [G*k]

	if self.useEIneurons:
		# ---------- 5. write to weight matrix (sparse or dense) ------------------
		half = cfg.hiddenLayerSize // 2
		excMask = newRows < half						# [G] bool
		inhMask = ~excMask

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
			rowsRel  = rowsSel + rowShift

			if mat.is_sparse:
				idx_new = torch.stack([rowsRel, colsSel], dim=0)
				val_new = valsSel #dtype is bool or float32
				mat = torch.sparse_coo_tensor(torch.cat([mat.indices(), idx_new], dim=1), torch.cat([mat.values(),  val_new]), size=mat.size(), device=device,).coalesce()
			else:
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
						if self.useEIneurons: # context of merge_into being called
							mat[rowsRel, colsSel] = valsSel.to(torch.int8) # True -> 1
						else:
							mat[rowsRel, colsSel] = torch.where(valsSel, 
							                                    torch.tensor(1, device=device, dtype=torch.int8), 
							                                    torch.tensor(-1, device=device, dtype=torch.int8))
					else:
						mat[rowsRel, colsSel] = valsSel.to(torch.int8) # direct cast if already numeric
				else:
					mat[rowsRel, colsSel] = valsSel
			return mat

		# split flattened arrays once for efficiency
		isExcEntry = flatRows < half
		flatRows_exc = flatRows[isExcEntry]
		flatCols_exc = flatCols[isExcEntry]
		flatVals_exc = flatVals[isExcEntry]

		flatRows_inh = flatRows[~isExcEntry]
		flatCols_inh = flatCols[~isExcEntry]
		flatVals_inh = flatVals[~isExcEntry]

		# merge into respective matrices
		Emat = self.hiddenConnectionMatrixExcitatory[layerIdx]
		Imat = self.hiddenConnectionMatrixInhibitory[layerIdx]

		Emat = merge_into(Emat, flatRows_exc, flatCols_exc, flatVals_exc, True)
		Imat = merge_into(Imat, flatRows_inh, flatCols_inh, flatVals_inh, False)

		self.hiddenConnectionMatrixExcitatory[layerIdx] = Emat
		self.hiddenConnectionMatrixInhibitory[layerIdx] = Imat
	else:
		# ---------- 5. write to weight matrix (sparse or dense) ------------------
		mat = self.hiddenConnectionMatrix[layerIdx]

		if mat.is_sparse:
			idx_new = torch.stack([flatRows, flatCols], dim=0)	   # [2, G*k]
			val_new = flatVals #dtype is bool or float32
			mat_new = torch.sparse_coo_tensor(torch.cat([mat.indices(), idx_new], dim=1), torch.cat([mat.values(),  val_new]), size=mat.size(), device=device,).coalesce()
		else:	# dense case
			mat_new = mat.clone()
			# Ensure flatVals is compatible with int8 matrix
			if mat_new.dtype == torch.int8:
				# flatVals can be bool (from sparse logic) or float (potentially, though aiming for int8)
				# This part handles the case where dense matrices are int8
				if flatVals.dtype == torch.bool:
					# Convert boolean flatVals to int8: True to 1, False to -1 (standard) or True to 1 (EI)
					if self.useEIneurons: # Context of the broader function
						# This assumes that for EI, weights are always 1. If flatVals can be False for EI, this needs adjustment.
						mat_new[flatRows, flatCols] = flatVals.to(torch.int8) # True -> 1
					else:
						mat_new[flatRows, flatCols] = torch.where(flatVals, 
						                                        torch.tensor(1, device=device, dtype=torch.int8), 
						                                        torch.tensor(-1, device=device, dtype=torch.int8))
				else:
					mat_new[flatRows, flatCols] = flatVals.to(torch.int8) # direct cast if already numeric
			else:
				mat_new[flatRows, flatCols] = flatVals

		# ---------- 6. store back -------------------------------------------------
		self.hiddenConnectionMatrix[layerIdx] = mat_new.to(device) #ensure device consistency


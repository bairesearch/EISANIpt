"""EISANIpt_EISANImodelPrune.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model prune (output connections and hidden neurons)

limitation: implementation currently requires useConnectionWeights

"""

import torch
import numpy as np
from ANNpt_globalDefs import *


def executePostTrainPrune(self, trainOrTest) -> None:
	if(trainOrTest):

		if debugMeasureClassExclusiveNeuronRatio:
			measure_class_exclusive_neuron_ratio(self)
		if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
			measure_ratio_of_hidden_neurons_with_output_connections(self)

		if limitOutputConnections:
			print("prune_output_connections_and_hidden_neurons():")
			prune_output_connections_and_hidden_neurons(self)

			if debugMeasureClassExclusiveNeuronRatio:
				measure_class_exclusive_neuron_ratio(self)
			if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
				measure_ratio_of_hidden_neurons_with_output_connections(self)

		if limitHiddenConnections:
			print("prune_hidden_neurons_and_output_connections():")
			prune_hidden_neurons_and_output_connections(self)

			if debugMeasureClassExclusiveNeuronRatio:
				measure_class_exclusive_neuron_ratio(self)
			if debugMeasureRatioOfHiddenNeuronsWithOutputConnections:
				measure_ratio_of_hidden_neurons_with_output_connections(self)

if limitOutputConnections:

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
	    	- self.outputConnectionMatrix is [Lhidden][hidden, C]
			
		current limitation: useSparseOutputMatrix does not support individual connection pruning via limitOutputConnections:limitOutputConnectionsBasedOnPrevalence (only hidden neuron pruning)
		"""

		# -------------------------------------------------------------------------
		# Helper 1: connection-level keep-mask (dense -> bool[H,C], CSR -> bool[nnz])
		# -------------------------------------------------------------------------
		def _keep_mask(hiddenLayerIdx: int, weights: torch.Tensor) -> torch.Tensor:
			
			if(useSparseOutputMatrix):
				crow, col, val = weights.crow_indices(), weights.col_indices(), weights.values()
				rows = weights.size(0)

				# Prevalence
				if limitOutputConnectionsBasedOnPrevalence and not useBinaryOutputConnectionsEffective and val.dtype != torch.bool:
					prevalent = val > limitOutputConnectionsPrevalenceMin
				else:
					prevalent = torch.ones_like(val, dtype=torch.bool)

				# Exclusivity (row_nnz == 1  \u2192 keep whole row)
				if limitOutputConnectionsBasedOnExclusivity:
					row_nnz = crow[1:] - crow[:-1]					# (rows,)
					excl_row = (row_nnz == 1)
					row_rep  = torch.repeat_interleave(torch.arange(rows, device=device), row_nnz)
					exclusive = excl_row[row_rep]
				else:
					exclusive = torch.ones_like(val, dtype=torch.bool)

				# Accuracy
				if limitOutputConnectionsBasedOnAccuracy:
					acc_layer = self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx][:,0].float() / self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx][:,1].clamp(min=1).float()
					acc_pass  = acc_layer > limitOutputConnectionsAccuracyMin			# (rows,)
					row_rep   = torch.repeat_interleave(torch.arange(rows, device=device), crow[1:] - crow[:-1])
					accurate  = acc_pass[row_rep]
				else:
					accurate = torch.ones_like(val, dtype=torch.bool)

				mask = prevalent & exclusive & accurate				# bool[nnz]
				print("mask.shape = ", mask.shape)
				return mask		# bool[nnz]		#explanation of [nnz] vs [H,C]: useSparseOutputMatrix supports hidden neuron pruning only (not individual output connection pruning)
			else:
				# ---- Dense path (original semantics) -------------------------
				mask = torch.ones_like(weights, dtype=torch.bool)
				if limitOutputConnectionsBasedOnPrevalence:
					prevalent = (weights > limitOutputConnectionsPrevalenceMin)
					mask &= prevalent
				if limitOutputConnectionsBasedOnExclusivity:
					exclusive = (weights.sum(dim=1, keepdim=True) == 1)
					mask &= exclusive
				if limitOutputConnectionsBasedOnAccuracy:
					acc_layer = self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx][:,0].float() /  self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx][:,1].clamp(min=1).float()
					accurate = (acc_layer > limitOutputConnectionsAccuracyMin).unsqueeze(1).expand_as(weights)
					mask &= accurate
				#print("mask.shape = ", mask.shape)
				return mask		# bool[H,C]

		# -------------------------------------------------------------------------
		# Helper 2: in-place "set kept" (dense) or rebuild (CSR)
		# -------------------------------------------------------------------------
		def _set_kept(mat: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:	# returns updated matrix
			if(useSparseOutputMatrix):
				# ---- CSR path ---------------------------------------------------
				coo = mat.to_sparse_coo()
				idx = coo.indices()					# (2, nnz)
				val = coo.values()					# (nnz,)
				idx_kept = idx[:, keep]
				val_kept = val[keep]

				new_coo = torch.sparse_coo_tensor(idx_kept, val_kept, size=mat.shape, device=mat.device, dtype=mat.dtype)
				mat = new_coo.to_sparse_csr()
			else:
				if useBinaryOutputConnectionsEffective:
					mat[...] = keep
				else:
					mat.mul_(keep.to(mat.dtype))
			return mat

		# -------------------------------------------------------------------------
		# Helper 3: neurons still feeding upper layers -> bool[hidden]
		# -------------------------------------------------------------------------
		def _still_used(hiddenLayerIdx: int) -> torch.Tensor:
			H = self._getCurrentHiddenLayerSize(hiddenLayerIdx)
			used = torch.zeros(H, dtype=torch.bool, device=device)
			if(useSequentialSANI):
				used = _connected_parents_mask(self, hiddenLayerIdx, 0) | _connected_parents_mask(self, hiddenLayerIdx, 1)
			else:
				for segmentIdx in range(numberOfSegmentsPerNeuron):
					if useEIneurons:
						mats = (self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx+1][segmentIdx], self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx+1][segmentIdx])
					else:
						mats = (self.hiddenConnectionMatrix[hiddenLayerIdx+1][segmentIdx],)
					for m in mats:
						if useSparseHiddenMatrix:
							prev_idx = m.coalesce().indices()[1]  # second dim indexes previous layer
							used[prev_idx.unique()] = True
						else:
							used |= (m != 0).any(dim=0)	#first dim indexes next layer
			return used

		# --------------------------------------------------------------------- #
		# Helper - slice out ONLY the assigned part and drop any "-1" padding
		# --------------------------------------------------------------------- #
		def _valid_parent_indices(self, hiddenLayerIdx: int, segmentIdx: int) -> torch.Tensor:
			used = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
			indices = (self.indexArrayA if segmentIdx == 0 else self.indexArrayB)[hiddenLayerIdx][:used]
			return indices[indices >= 0]			# safeguard (should already be non-negative)

		# --------------------------------------------------------------------- #
		# Boolean mask of parents used by segment-X at `hiddenLayerIdx`
		# --------------------------------------------------------------------- #
		def _connected_parents_mask(self, hiddenLayerIdx: int, segmentIdx: int) -> torch.Tensor:
			"""
			Returns BoolTensor[ prevLayerSize ] \u2013 True where the neuron *is* wired into
			segment-A of layer `layerIdx`.
			"""
			prevSize = self._getCurrentHiddenLayerSize(hiddenLayerIdx)
			mask = torch.zeros(prevSize, dtype=torch.bool, device=device)
			indices = _valid_parent_indices(self, hiddenLayerIdx+1, segmentIdx)
			if indices.numel() > 0:
				mask.scatter_(0, indices, True)		# vectorised, in-place
			#print("_connected_parents_mask: mask.shape = ", mask.shape)
			return mask

		# -------------------------------------------------------------------------
		# Core logic (same top/bottom layout as original)
		# -------------------------------------------------------------------------
		if useOutputConnectionsLastLayer:
			# -------- Case A: only final hidden layer owns output connections ---
			hiddenLayerIdx = self.numberUniqueHiddenLayers - 1
			oc             = self.outputConnectionMatrix[hiddenLayerIdx]

			keep = _keep_mask(hiddenLayerIdx, oc)					# bool[H,C] or bool[nnz]
			
			if(not useSequentialSANI):
				# apply pruning mask to output connections (zero out only)
				if useSparseOutputMatrix:
					oc = _set_kept(oc, keep)
					self.outputConnectionMatrix[hiddenLayerIdx] = oc
				else:
					_set_kept(oc, keep)

			#identify dead hidden neurons
			if useSparseOutputMatrix:
				rm_mask = (get_row_nnz(oc) == 0)
			else:
				rm_mask = ~(oc != 0).any(dim=1)
			rm_mask = filterByAssignedMask(self, hiddenLayerIdx, rm_mask)

			pruneHiddenNeurons(self, hiddenLayerIdx, rm_mask)

			if(useSequentialSANI):
				#apply pruning mask to output connections - actual outputConnectionMatrix shape is modified
				self.outputConnectionMatrix[hiddenLayerIdx] = prune_output_rows(self.outputConnectionMatrix[hiddenLayerIdx], rm_mask)
		else:
			# -------- Case B: every hidden layer owns output connections --------
			for hiddenLayerIdx in reversed(range(self.numberUniqueHiddenLayers)):
				oc_layer = self.outputConnectionMatrix[hiddenLayerIdx]

				keep = _keep_mask(hiddenLayerIdx, oc_layer)		# bool[H,C] or bool[nnz]

				# ---- ensure neurons feeding upper layers survive --------------
				if hiddenLayerIdx < self.numberUniqueHiddenLayers - 1:
					row_keep = _still_used(hiddenLayerIdx)			# bool[H]
					if useSparseOutputMatrix:
						# Map row_keep to nnz-level and OR with keep
						crow = oc_layer.crow_indices()
						row_counts = crow[1:] - crow[:-1]
						row_rep = torch.repeat_interleave(torch.arange(row_keep.size(0), device=device), row_counts)
						keep |= row_keep[row_rep]
					else:
						keep |= row_keep.unsqueeze(1)

				if(not useSequentialSANI):
					# apply pruning mask to output connections (zero out only)
					if useSparseOutputMatrix:
						oc_layer = _set_kept(oc_layer, keep)
						self.outputConnectionMatrix[hiddenLayerIdx] = oc_layer
					else:
						_set_kept(oc_layer, keep)

				#identify dead hidden neurons
				if useSparseOutputMatrix:
					rm_mask = (get_row_nnz(oc_layer) == 0)
				else:
					rm_mask = ~(oc_layer != 0).any(dim=1)
				rm_mask = filterByAssignedMask(self, hiddenLayerIdx, rm_mask)

				pruneHiddenNeurons(self, hiddenLayerIdx, rm_mask)
				
				if(useSequentialSANI):
					#apply pruning mask to output connections - actual outputConnectionMatrix shape is modified
					self.outputConnectionMatrix[hiddenLayerIdx] = prune_output_rows(self.outputConnectionMatrix[hiddenLayerIdx], rm_mask)


if limitHiddenConnections:

	# ================================================================================
	# Usage-based pruning of hidden neurons *and* their output connections
	# ================================================================================
	def prune_hidden_neurons_and_output_connections(self) -> None:
		"""
		Prune every hidden neuron whose recorded usage is **below**
		``hiddenNeuronMinUsageThreshold`` and drop its outgoing
		output-layer weights.

		The heavy lifting (mask bookkeeping, signature purging, segment
		resetting...) is delegated to :func:`pruneHiddenNeurons`, so all
		internal structures stay consistent.
		"""
		#reversed is required for useSequentialSANI only (complete neuron deletion; not simply zero out)
		for hiddenLayerIdx in reversed(range(self.numberUniqueHiddenLayers)):
			if(limitHiddenConnectionsBasedOnPrevalence):
				usage_layer = self.hiddenNeuronUsage[hiddenLayerIdx]				# (hidden,)
				#print("prune_hidden_neurons_and_output_connections: hiddenLayerIdx = ", hiddenLayerIdx)
				#print("usage_layer.shape = ", usage_layer.shape)
				#print("self.outputConnectionMatrix[hiddenLayerIdx].shape = ", self.outputConnectionMatrix[hiddenLayerIdx].shape)
				rm_mask = (usage_layer < hiddenNeuronMinUsageThreshold)
				rm_mask = filterByAssignedMask(self, hiddenLayerIdx, rm_mask)
				if not rm_mask.any():
					continue
			else:
				printe("prune_hidden_neurons_and_output_connections currently requires limitHiddenConnectionsBasedOnPrevalence")

			# ---- 1. prune output-connection rows ----------------------------
			if useOutputConnectionsLastLayer:
				# only the topmost hidden layer owns the matrix
				if hiddenLayerIdx == self.numberUniqueHiddenLayers - 1:
					self.outputConnectionMatrix = prune_output_rows(self.outputConnectionMatrix[hiddenLayerIdx], rm_mask)
			else:
				self.outputConnectionMatrix[hiddenLayerIdx] = prune_output_rows(self.outputConnectionMatrix[hiddenLayerIdx], rm_mask)

			# ---- 2. prune hidden neuron + ancillary data --------------------
			pruneHiddenNeurons(self, hiddenLayerIdx, rm_mask)

			# ---- 3. zero usage so repeated calls behave predictably ---------
			usage_layer = 0	#usage_layer[rm_mask] = 0

if(useSequentialSANI):

	# ----------------------------------------------------------
	# b) Output-row pruning (dense or CSR)
	# ----------------------------------------------------------
	def prune_output_rows(outputConnectionMat: torch.Tensor, hiddenNeuronsRemoved: torch.Tensor) -> torch.Tensor:
		"""
		Return a *new* matrix where every row whose corresponding neuron is
		marked for removal has been physically deleted.

		Supports 2-D dense tensors and CSR tensors created with
		``torch.sparse_csr_tensor``.
		"""
		if not hiddenNeuronsRemoved.any():
			return outputConnectionMat					# unchanged

		keep_mask	= ~hiddenNeuronsRemoved
		n_rows_new	= int(keep_mask.sum())
		n_classes	= outputConnectionMat.size(1)

		if(useSparseOutputMatrix):
			# -- sparse CSR ------------------------------------------------------------
			# Convert to COO for easy row filtering, then back to CSR.
			coo			= outputConnectionMat.to_sparse_coo()
			row_idx, col_idx = coo.indices()
			vals		= coo.values()

			keep_entry_mask		= keep_mask[row_idx]			# keep only surviving rows
			row_idx_kept		= row_idx[keep_entry_mask]
			col_idx_kept		= col_idx[keep_entry_mask]
			vals_kept			= vals[keep_entry_mask]

			# Map old -> new row indices (compressed)
			new_row_map			= torch.full_like(keep_mask, -1, dtype=torch.long)
			new_row_map[keep_mask] = torch.arange(n_rows_new, device=keep_mask.device)
			row_idx_new			= new_row_map[row_idx_kept]

			# Build new COO -> CSR
			new_indices			= torch.stack([row_idx_new, col_idx_kept])
			new_coo				= torch.sparse_coo_tensor(
				new_indices, vals_kept,
				size=(n_rows_new, n_classes),
				device=outputConnectionMat.device,
				dtype=outputConnectionMat.dtype
			)
			result = new_coo.to_sparse_csr()
		else:
			result = outputConnectionMat[keep_mask]		# (n_rows_new, classes)
			
		return result

	# ----------------------------------------------------------
	# a) Hidden-neuron pruning
	# ----------------------------------------------------------
	def pruneHiddenNeurons(self, hiddenLayerIdx: int, hiddenNeuronsRemoved: torch.Tensor) -> None:
		"""
		Completely delete (not just zero-out) hidden neurons flagged in
		``hiddenNeuronsRemoved`` and keep every per-layer tensor in sync.

		Parameters
		----------
		hiddenLayerIdx : int
			Which *unique* hidden layer is being pruned.
		hiddenNeuronsRemoved : BoolTensor, shape = [hidden]
			True wherever a neuron must be removed.
		"""
		printPruneHiddenNeurons(self, hiddenLayerIdx, hiddenNeuronsRemoved)

		if not hiddenNeuronsRemoved.any():
			return										# nothing to do

		# ---- masks & bookkeeping -------------------------------------------------
		keep_mask		= ~hiddenNeuronsRemoved			# neurons to retain
		n_removed		= int(hiddenNeuronsRemoved.sum())

		# Update segment-count tracker (never let it drop below zero)
		self.numAssignedNeuronSegments[hiddenLayerIdx] = torch.clamp(self.numAssignedNeuronSegments[hiddenLayerIdx] - n_removed, min=0,)

		# ---- shrink per-layer arrays --------------------------------------------
		self.indexArrayA[hiddenLayerIdx] = self.indexArrayA[hiddenLayerIdx][keep_mask]
		self.indexArrayB[hiddenLayerIdx] = self.indexArrayB[hiddenLayerIdx][keep_mask]

		# Prune usage statistics if they exist
		self.hiddenNeuronUsage[hiddenLayerIdx] = self.hiddenNeuronUsage[hiddenLayerIdx][keep_mask]

		# Any other per-neuron tensors can be added here in the same one-liner style
		# (e.g. self.neuronSegmentAssignedMask[...] = ...[keep_mask])
		if(limitOutputConnections and limitOutputConnectionsBasedOnAccuracy):
			self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx] = self.hiddenNeuronPredictionAccuracy[hiddenLayerIdx][keep_mask]

else:
	# ------------------------------------------------------------------
	# Wrapper that chooses the correct row-clearing strategy
	# ------------------------------------------------------------------
	def prune_output_rows(mat: torch.Tensor, rm_mask: torch.Tensor) -> torch.Tensor:
		if(useSparseOutputMatrix):
			return zero_rows_sparse_csr(mat, rm_mask)
			#return prune_rows(mat, rm_mask)			# COO path
		else:
			mat[rm_mask] = 0
			return mat

	# ------------------------------------------------------------------
	# CSR helper - clear all data in the rows flagged by *rm_mask*
	# ------------------------------------------------------------------
	def zero_rows_sparse_csr(mat: torch.Tensor, rm_mask: torch.Tensor) -> torch.Tensor:
		"""
		Return a new ``torch.sparse_csr_tensor`` with the *same shape* as *mat*
		but with all entries in rows where ``rm_mask == True`` removed.
		"""
		crow = mat.crow_indices()
		col  = mat.col_indices()
		val  = mat.values()

		rows = rm_mask.shape[0]
		new_crow = torch.empty_like(crow)
		new_col_chunks = []
		new_val_chunks = []

		nnz_accum = 0
		new_crow[0] = 0
		for r in range(rows):
			start = crow[r].item()
			end   = crow[r + 1].item()
			if rm_mask[r]:
				length = 0					# discard
			else:
				length = end - start
				if length:
					new_col_chunks.append(col[start:end])
					new_val_chunks.append(val[start:end])
			nnz_accum += length
			new_crow[r + 1] = nnz_accum

		if new_col_chunks:
			new_col = torch.cat(new_col_chunks)
			new_val = torch.cat(new_val_chunks)
		else:
			new_col = col.new_empty((0,))
			new_val = val.new_empty((0,))

		return torch.sparse_csr_tensor(new_crow, new_col, new_val, size=mat.shape, dtype=mat.dtype, device=mat.device)

	def pruneHiddenNeurons(self, hiddenLayerIdx: int, hiddenNeuronsRemoved: torch.Tensor) -> None:
		"""
		Delete (or zero-out) hidden neurons whose *output* fan-out is now zero.

		Arguments
		---------
		hiddenLayerIdx : int
	    	Index of the unique hidden layer being pruned.
		hiddenNeuronsRemoved : torch.BoolTensor  shape = [hidden]
	    	True for every neuron that must disappear.

		Note prunes all segments
		"""
		printPruneHiddenNeurons(self, hiddenLayerIdx, hiddenNeuronsRemoved)
		
		if not hiddenNeuronsRemoved.any():
			return	# no work

		dev = hiddenNeuronsRemoved.device
		nSeg = numberOfSegmentsPerNeuron
		H_total = self.config.hiddenLayerSize

		# ---- prune hidden - hidden weight matrices ----
		for segmentIdx in range(numberOfSegmentsPerNeuron): 

			if useEIneurons:
				if useInhibition:
					H_exc = H_total // 2
					ex_rm = hiddenNeuronsRemoved[:H_exc]
					ih_rm = hiddenNeuronsRemoved[H_exc:]
					self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIdx] = prune_rows(self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx][segmentIdx], ih_rm)
				else:
					H_exc = H_total  # All neurons are excitatory
					ex_rm = hiddenNeuronsRemoved

				self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIdx] = prune_rows(self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx][segmentIdx], ex_rm)
			else:
				self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIdx] = prune_rows(self.hiddenConnectionMatrix[hiddenLayerIdx][segmentIdx], hiddenNeuronsRemoved)

			# ---- dynamic-growth bookkeeping -----------------------------------
			if useDynamicGeneratedHiddenConnections:
				self.neuronSegmentAssignedMask[hiddenLayerIdx, segmentIdx, hiddenNeuronsRemoved] = False

			if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
				if useEIneurons:
					if useInhibition:
						H_exc = H_total // 2
						purge_sigs(self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIdx], hiddenNeuronsRemoved[:H_exc], H_total)
						purge_sigs(self.hiddenNeuronSignaturesInh[hiddenLayerIdx][segmentIdx], hiddenNeuronsRemoved[H_exc:], H_total)
					else:
						purge_sigs(self.hiddenNeuronSignaturesExc[hiddenLayerIdx][segmentIdx], hiddenNeuronsRemoved, H_total)
				else:
					purge_sigs(self.hiddenNeuronSignatures[hiddenLayerIdx][segmentIdx], hiddenNeuronsRemoved, H_total)

		# Prune usage statistics if they exist
		self.hiddenNeuronUsage[hiddenLayerIdx][hiddenNeuronsRemoved] = 0


	def prune_rows(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		"""Return matrix with rows in *mask* removed (sparse) or zeroed (dense)."""
		#note that sparse tensors do not store zeroed out values
		if(useSparseHiddenMatrix):
			mat = mat.coalesce()
			idx = mat.indices()
			val = mat.values()
			keep = ~mask[idx[0]]
			return torch.sparse_coo_tensor(idx[:, keep], val[keep], size=mat.shape, dtype=mat.dtype, device=mat.device).coalesce()
		else:
			mat[mask] = 0
			return mat

	def purge_sigs(seg, rm_mask, H_total):
		dict_seg = seg
		to_del   = []
		for sig, nid in dict_seg.items():
			idx = int(nid.item() if isinstance(nid, torch.Tensor) else nid)
			if idx < H_total and rm_mask[idx].item():
				to_del.append(sig)
		for sig in to_del:
			dict_seg.pop(sig, None)


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def get_row_nnz(mat: torch.Tensor) -> torch.Tensor:
	"""
	Return a 1-D LongTensor giving the number of non-zero elements
	(~= connected classes) in every row of ``mat``.

	Works for:
	* Dense 2-D tensors
	* CSR tensors created with torch.sparse_csr_tensor
	"""
	if useSparseOutputMatrix:
		# CSR:  crow[i] <= ... < crow[i+1]  -> nnz = diff(crow)
		crow = mat.crow_indices()					# (rows+1,)
		return crow[1:] - crow[:-1]					# (rows,)
	else:
		# Dense
		return (mat != 0).sum(dim=1)				# (rows,)


# ----------------------------------------------------------
# Ratio "class-exclusive / non-exclusive"
# ----------------------------------------------------------
def measure_class_exclusive_neuron_ratio(self) -> float:
	"""
	Compute ratio of class-exclusive to non-class-exclusive hidden neurons:
		Ratio of *class-exclusive* neurons (exactly one connected class) to *non-exclusive* neurons (>1 connected class).
	"""
	ratio_all_layers	= 0.0
	layer_cnt			= 0

	for hl_idx in range(self.config.numberOfHiddenLayers):
		if not useOutputConnectionsLastLayer or hl_idx == self.config.numberOfHiddenLayers - 1:
			oc = self.outputConnectionMatrix[hl_idx]
			if oc is None:
				continue
			row_nnz = get_row_nnz(oc)					# (hidden,)
			if row_nnz.numel() == 0:
				continue
			'''
			print("oc = ", oc)
			print("row_nnz = ", row_nnz)
			np.savetxt("oc.csv", oc.cpu().numpy(), delimiter=",", fmt="%.6f")
			np.savetxt("row_nnz.csv", row_nnz.cpu().numpy(), delimiter=",", fmt="%.6f")
			'''
			exclusive		= (row_nnz == 1).sum().item()
			non_exclusive	= (row_nnz > 1).sum().item()
			ratio = float('inf') if non_exclusive == 0 else exclusive / non_exclusive
			printf("\tmeasure_class_exclusive_neuron_ratio:hl_idx = ", hl_idx, ", ratio = ", ratio, ", exclusive = ", exclusive, ", non_exclusive = ", non_exclusive)
			ratio_all_layers += ratio
			layer_cnt += 1

	if layer_cnt == 0:
		return 0.0
	ratio_all_layers /= layer_cnt
	printf("measure_class_exclusive_neuron_ratio:ratioAllLayers = ", ratio_all_layers)
	return ratio_all_layers


# ----------------------------------------------------------
# Ratio "has "1 output connection"
# ----------------------------------------------------------
def measure_ratio_of_hidden_neurons_with_output_connections(self) -> float:
	"""
	Compute ratio of hidden neurons having any output connection:
		Ratio of hidden neurons that connect to at least one class.
	"""
	ratio_all_layers	= 0.0
	layer_cnt			= 0

	for hl_idx in range(self.config.numberOfHiddenLayers):
		if not useOutputConnectionsLastLayer or hl_idx == self.config.numberOfHiddenLayers - 1:
			oc = self.outputConnectionMatrix[hl_idx]
			if oc is None:
				continue							# might happen when useOutputConnectionsLastLayer=True
			row_nnz = get_row_nnz(oc)				# (hidden,)
			if row_nnz.numel() == 0:
				continue
			any_conn	= (row_nnz > 0)
			ratio		= any_conn.sum().item() / any_conn.numel()
			ratio_all_layers += ratio
			printf("\tmeasure_ratio_of_hidden_neurons_with_output_connections:hl_idx = ", hl_idx, ", ratio = ", ratio)
			layer_cnt += 1

	if layer_cnt == 0:
		return 0.0									# no layers examined
	ratio_all_layers /= layer_cnt
	printf("measure_ratio_of_hidden_neurons_with_output_connections:ratioAllLayers = ", ratio_all_layers)
	return ratio_all_layers

def printPruneHiddenNeurons(self, hiddenLayerIdx, hiddenNeuronsRemoved):
	if(printLimitConnections):
		total_neurons = hiddenNeuronsRemoved.numel()                   # hiddenLayerSizeSANI
		if(useSequentialSANI):
			assigned_count = self.numAssignedNeuronSegments[hiddenLayerIdx].item()
			removed_count = hiddenNeuronsRemoved.sum().item()
		else:
			assigned_mask = self.neuronSegmentAssignedMask[hiddenLayerIdx].any(dim=0)	#does not properly support multiple segments
			assigned_count = assigned_mask.sum().item()
			removed_count = hiddenNeuronsRemoved.sum().item()      # intersection
			'''
			#redundant now with filterByAssignedMask(hiddenLayerIdx, rm_mask);
			removed_assigned_mask = hiddenNeuronsRemoved & assigned_mask
			removed_count = removed_assigned_mask.sum().item()      # intersection
			'''
		perc_total = removed_count / total_neurons  * 100.0
		perc_assigned = (removed_count / assigned_count * 100.0) if assigned_count else 0.0
		printf("pruneHiddenNeurons: layer=", hiddenLayerIdx, ", removed=", removed_count, "/ assigned=", assigned_count, "/ hiddenLayerSizeSANI=", total_neurons, "(", round(perc_assigned, 2), "% of assigned;", round(perc_total,   2), "% of all)")  

def filterByAssignedMask(self, hiddenLayerIdx, rm_mask):
	if(useSequentialSANI):
		assigned_mask = torch.arange(self._getCurrentHiddenLayerSize(hiddenLayerIdx), device=device) < self.numAssignedNeuronSegments[hiddenLayerIdx]
	else:
		assigned_mask = self.neuronSegmentAssignedMask[hiddenLayerIdx].any(dim=0)	#does not properly support multiple segments
	rm_mask = rm_mask & assigned_mask
	return rm_mask
			

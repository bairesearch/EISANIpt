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

"""

import torch
from ANNpt_globalDefs import *

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
	if not useOutputConnectionsLastLayer:
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
	if not useOutputConnectionsLastLayer:
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
	    - when useOutputConnectionsLastLayer == True: self.outputConnectionMatrix is [hidden,   C]
	    - when useOutputConnectionsLastLayer == False: self.outputConnectionMatrix is [Lhidden, hidden, C]
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

		for upper in range(layer_idx + 1, self.numberUniqueHiddenLayers):
			if useEIneurons:
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

	if useOutputConnectionsLastLayer:
		# -------- Case A: only final hidden layer owns output connections ---
		oc   = self.outputConnectionMatrix                     # [hidden, C]
		keep = _keep_mask(self.numberUniqueHiddenLayers-1, oc)
		_set_kept(oc, keep)

		removed = ~(oc != 0).any(dim=1)                        # neurons now dead
		pruneHiddenNeurons(self, self.numberUniqueHiddenLayers - 1, removed)
	else:
		# -------- Case B: every hidden layer owns output connections --------
		for l in reversed(range(self.numberUniqueHiddenLayers)):
			oc_layer = self.outputConnectionMatrix[l]              # [hidden, C]
			keep = _keep_mask(l, oc_layer)

			if l < self.numberUniqueHiddenLayers - 1:                    # not topmost
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
	nSeg    = numberOfSegmentsPerNeuron
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
	if useEIneurons:
		if useInhibition:
			H_exc = H_total // 2
			ex_rm = hiddenNeuronsRemoved[:H_exc]
			ih_rm = hiddenNeuronsRemoved[H_exc:]
			self.hiddenConnectionMatrixInhibitory[layerIndex] = _prune_rows(self.hiddenConnectionMatrixInhibitory[layerIndex], ih_rm)
		else:
			H_exc = H_total  # All neurons are excitatory
			ex_rm = hiddenNeuronsRemoved

		self.hiddenConnectionMatrixExcitatory[layerIndex] = _prune_rows(self.hiddenConnectionMatrixExcitatory[layerIndex], ex_rm)
	else:
		self.hiddenConnectionMatrix[layerIndex] = _prune_rows(self.hiddenConnectionMatrix[layerIndex], hiddenNeuronsRemoved)

	# ---- dynamic-growth bookkeeping -----------------------------------
	if useDynamicGeneratedHiddenConnections:
		self.neuronSegmentAssignedMask[layerIndex, hiddenNeuronsRemoved, :] = False

	if useDynamicGeneratedHiddenConnectionsUniquenessChecks:
		if useEIneurons:
			if useInhibition:
				H_exc = H_total // 2
				_purge_sigs(self.hiddenNeuronSignaturesExc[layerIndex], hiddenNeuronsRemoved[:H_exc])
				_purge_sigs(self.hiddenNeuronSignaturesInh[layerIndex], hiddenNeuronsRemoved[H_exc:])
			else:
				_purge_sigs(self.hiddenNeuronSignaturesExc[layerIndex], hiddenNeuronsRemoved)
		else:
			_purge_sigs(self.hiddenNeuronSignatures[layerIndex], hiddenNeuronsRemoved)

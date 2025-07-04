"""EISANIpt_EISANImodelOutput.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model output

"""

import torch
from ANNpt_globalDefs import *
import torch.nn.functional as F
if(limitConnections):
	from EISANIpt_EISANImodelPrune import get_row_nnz

# -----------------------------
# Output layer
# -----------------------------

def calculateOutputLayer(self, trainOrTest, layerActivations, y):
	outputActivations = torch.zeros(self.batchSize, self.config.numberOfClasses, device=device)
	assert y is not None
	
	actLayerIndex = 0
	for layerIdSuperblock in range(recursiveSuperblocksNumber):
		for layerIdHidden in range(self.config.numberOfHiddenLayers):
			isLastLayer = (layerIdSuperblock==recursiveSuperblocksNumber-1) and (layerIdHidden==self.config.numberOfHiddenLayers-1)
			if(not useOutputConnectionsLastLayer or isLastLayer):
				act = layerActivations[actLayerIndex]
				uniqueLayerIndex = self._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden)
				
				if(not generateConnectionsAfterPropagating):
					# Training: reinforce output connections
					if trainOrTest or evalStillTrainOutputConnections:
						update_output_connections(self, uniqueLayerIndex, act, y, device)
				
				weights = self.outputConnectionMatrix[uniqueLayerIndex]
					
				if(debugEISANIoutputs):
					print("\tuniqueLayerIndex = ", uniqueLayerIndex)
					print("layerActivations[actLayerIndex].shape = ", layerActivations[actLayerIndex].shape)
					print("self.outputConnectionMatrix[uniqueLayerIndex].shape = ", self.outputConnectionMatrix[uniqueLayerIndex].shape)
					row_ptr = self.outputConnectionMatrix[uniqueLayerIndex].crow_indices()      # shape = (nrows+1,)
					total_nnz = int(row_ptr[-1].item())
					print("total output connections = ", total_nnz)
		
				if useBinaryOutputConnectionsEffective:
					weights = weights.to(torch.bool).to(torch.int8)	# float to bool to int8 (0/1)
				else:
					if useBinaryOutputConnections:
						weights = weights.to(torch.int8) # bool to int8 (0/1)
					else:
						weights = weights

				weights = normaliseOutputConnectionWeights(self, weights.float())	# cast weights to float for matmul
				
				if(useSparseOutputMatrix):
					outputActivations += act.float() @ weights 	# cast act to float for matmul
				else:
					outputActivations += act.float() @ weights 	# cast act to float for matmul
					
				#print("uniqueLayerIndex = ", uniqueLayerIndex)
				#print("act = ", act)	
				#print("outputActivations = ", outputActivations)
				
				if(generateConnectionsAfterPropagating):
					# Training: reinforce output connections
					if trainOrTest or evalStillTrainOutputConnections:
						update_output_connections(self, uniqueLayerIndex, act, y, device)
					
			actLayerIndex += 1

	predictions = torch.argmax(outputActivations, dim=1)
	#print("predictions = ", predictions)
	return predictions

def normaliseOutputConnectionWeights(self, weights):
	if(useOutputConnectionsNormalised):
		if(useSparseOutputMatrix):
			vals = weights.values().float()
			vals = torch.tanh(vals / useOutputConnectionsNormalisationRange)
			weights = torch.sparse_csr_tensor(weights.crow_indices(), weights.col_indices(), vals, size=weights.shape, device=weights.device)
		else:
			weights = torch.tanh(weights.float() / useOutputConnectionsNormalisationRange)
	return weights

def update_output_connections(self,
		hiddenLayerIdx: int,
		activation: torch.Tensor,		 # (B,H) bool / int8
		y: torch.Tensor,				  # (B,)   long
		device: torch.device) -> None:

	#orig before optimisation;

	mat = self.outputConnectionMatrix[hiddenLayerIdx]

	# -- gather all (row = neuron , col = class) pairs in one shot ---------
	sample_idx, neuron_idx = (activation != 0).nonzero(as_tuple=True)
	if neuron_idx.numel() == 0:
		return													# nothing active

	class_idx = y[sample_idx]									# align with rows

	if(useNLPDataset):
		# ----- drop pad-token targets ---------------------------------------
		valid_mask = (class_idx != NLPcharacterInputPadTokenID)
		sample_idx = sample_idx[valid_mask]
		neuron_idx = neuron_idx[valid_mask]
		class_idx  = class_idx[valid_mask]
		if neuron_idx.numel() == 0:
			return		

	if useBinaryOutputConnections or not self.useSequentialSANIactivationStrength:
		# keep binary semantics: any positive activation -> True
		new_vals = torch.ones_like(neuron_idx, dtype=torch.bool, device=device)
	else:
		# use the *actual* activation value for each (sample, neuron) hit
		new_vals = activation[sample_idx, neuron_idx].to(torch.float32)

	if(useSparseOutputMatrix):
		# 1)   CSR -> COO (cheap view, no copy)
		coo = mat.to_sparse_coo()

		# 2)   concatenate indices & values
		exist_idx = coo.indices()			# shape (2, nnz)
		exist_val = coo.values()

		new_idx   = torch.stack([neuron_idx, class_idx])		# (2, new)
		all_idx   = torch.cat([exist_idx, new_idx], dim=1)
		all_val   = torch.cat([exist_val, new_vals])

		# 3)   coalesce -> merges dups (adds values for float / int dtypes)
		coo_upd = torch.sparse_coo_tensor(all_idx, all_val, size=mat.shape, device=device, dtype=all_val.dtype).coalesce()

		# 4)   bool weights -> cast back to bool after OR-like merge
		if useBinaryOutputConnections:
			coo_upd = torch.sparse_coo_tensor(coo_upd.indices(), (coo_upd.values() != 0), size=coo_upd.shape, device=device, dtype=torch.bool)

		# 5)   back to CSR for fast mat-mul
		self.outputConnectionMatrix[hiddenLayerIdx] = coo_upd.to_sparse_csr()
	else:
		if useBinaryOutputConnections:
			mat[neuron_idx, class_idx] = True						# logical OR
		else:
			# -------- accumulate real activation strengths (B \u2265 1) -----------
			vals      = activation[sample_idx, neuron_idx].to(mat.dtype)
			flat_mat  = mat.view(-1)								# (H*C,)
			flat_idx  = neuron_idx * mat.shape[1] + class_idx		# linear indices
			flat_mat.index_put_((flat_idx,), vals, accumulate=True)


# -----------------------------------------------------------------
# Update hidden-neuron accuracy statistics (soft-max vote)
# -----------------------------------------------------------------

def updateHiddenNeuronAccuracyStatistics(self, trainOrTest, layerActivations, y):
	if(limitConnections and limitOutputConnections and limitOutputConnectionsBasedOnAccuracy):
		if y is not None:												# labels available
			#note if useOutputConnectionsLastLayer, only the last index in hiddenNeuronPredictionAccuracy will be valid

			for lidx, act in enumerate(layerActivations):

				#normalise output weights
				w_norm = normaliseOutputConnectionWeights(self, self.outputConnectionMatrix[lidx])
				
				if useSparseOutputMatrix:
					if(limitOutputConnectionsBasedOnAccuracySoftmax):
						soft_true = _csr_softmax_true_class_probs(w_norm, y)  # [B,H]
					else:
						soft_true = csr_select_columns(w_norm, y)   # [B, H]
				else:
					if(limitOutputConnectionsBasedOnAccuracySoftmax):
						w_soft = torch.softmax(w_norm, dim=-1)
						soft_true = w_soft[:, y].t()                           # [B,H]
					else:
						soft_true = w_norm[:, y].t()   
				pred_above_thr	= soft_true > limitOutputConnectionsSoftmaxWeightMin	# [B,H]

				row_nnz = get_row_nnz(self.outputConnectionMatrix[lidx])				# [H]
				conn_layer = (row_nnz > 0)				# (H,)	#orig (dense only): (self.outputConnectionMatrix[lidx] != 0).any(dim=1)
				valid_neuron_mask = conn_layer.unsqueeze(0)				# [1,H]
				
				active_mask = (act != 0)								# [B,H]

				correct_neuron = active_mask & valid_neuron_mask & pred_above_thr
				considered = active_mask & valid_neuron_mask

				self.hiddenNeuronPredictionAccuracy[lidx][:,0] += correct_neuron.sum(dim=0)
				self.hiddenNeuronPredictionAccuracy[lidx][:,1] += considered.sum(dim=0)


if(limitOutputConnectionsBasedOnAccuracySoftmax):
	# ----------------------------------------------------------
	# 2.  Gather true-class probabilities for a batch
	#     (sparse equivalent of w_soft[:, y].t())
	# ----------------------------------------------------------
	def _csr_softmax_true_class_probs(csr_mat: torch.Tensor, labels: torch.LongTensor):
		"""
		Parameters
		----------
		csr_mat : CSR tensor shape = (H, C)
		labels  : LongTensor shape = (B,)   -- ground-truth label for every sample

		Returns
		-------
		soft_true : Tensor float32 shape = (B, H)
	            	soft-max probability assigned by each hidden neuron (row)
	            	to the *true* class of every sample in the batch.
		"""
		# -- precompute soft-max for *all* nnz entries -----------------------
		row_idx, col_idx, prob_val, row_den = _row_softmax_probs_csr(csr_mat)

		H = csr_mat.size(0)
		B = labels.size(0)
		device = csr_mat.device
		dtype  = prob_val.dtype

		# Default probability for a class that has **no explicit connection**
		# in that row:   p = e^0 / denom = 1 / denom
		default_p = (1.0 / row_den).to(dtype)		# (H,)

		# We will overwrite these defaults wherever a (row, label) pair
		# actually exists in the sparse matrix.
		soft_true = default_p.expand(B, H).clone()	# (B, H)

		# --- put NNZ probabilities into soft_true ---------------------------
		# 1.  For each label value, grab the nnz entries that match it.
		#     This loop is tiny: it iterates over *batch size*, not classes�rows.
		for b, lab in enumerate(labels.tolist()):
			mask = (col_idx == lab)
			if mask.any():
				soft_true[b, row_idx[mask]] = prob_val[mask]

		return soft_true

	# ----------------------------------------------------------
	# 1.  Soft-max over a CSR matrix, nnz only
	# ----------------------------------------------------------
	def _row_softmax_probs_csr(csr_mat: torch.Tensor):
		"""
		Compute soft-max *per row* for a 2-D **CSR** tensor *without* converting it
		to dense.

		Returns
		-------
		row_idx : LongTensor (nnz,)   -- row index for every nnz entry
		col_idx : LongTensor (nnz,)   -- corresponding column index
		prob_val: Tensor     (nnz,)   -- soft-max probability of that entry
		row_den : Tensor     (rows,)  -- denominator of the soft-max for each row
		"""
		assert csr_mat.is_sparse_csr, "_row_softmax_probs_csr expects a CSR tensor"

		rows, classes = csr_mat.size()
		crow	= csr_mat.crow_indices()		# (rows+1,)
		col_idx	= csr_mat.col_indices()			# (nnz,)
		val		= csr_mat.values()				# (nnz,)

		# 1) exp() on all non-zero weights
		exp_val = torch.exp(val)

		# 2) build row_idx for every nnz entry (repeat_interleave is zero-copy)
		row_idx = torch.repeat_interleave(
			torch.arange(rows, device=val.device),
			crow[1:] - crow[:-1]
		)											# (nnz,)

		# 3) row-wise sum of exp(w)
		row_sum_nz = torch.zeros(rows, device=val.device).scatter_add_(0, row_idx, exp_val)

		# 4) number of *missing* columns per row and full denominator
		n_nz	= crow[1:] - crow[:-1]				# (rows,)
		row_den	= row_sum_nz + (classes - n_nz)		# + e^0 (=1) for every implicit zero

		# 5) probability for every nnz entry
		prob_val = exp_val / row_den[row_idx]

		return row_idx, col_idx, prob_val, row_den
else:

	# ----------------------------------------------------------
	# CSR column-selection without Python loops
	# ----------------------------------------------------------
	def csr_select_columns(w_csr: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
		"""
		Return a dense tensor equivalent to ``w_csr[:, labels].t()`` without
		converting the whole CSR matrix to dense and **without any Python loops**.

		Parameters
		----------
		w_csr  : CSR tensor, shape = (H, C), dtype = float / bool / \u2026
		labels : LongTensor, shape = (B,)      -- one class-index per sample

		Returns
		-------
		out : Tensor, shape = (B, H), dtype = w_csr.dtype
	    	  out[b, h] = w_csr[h, labels[b]]
		"""
		assert w_csr.is_sparse_csr, "expected a CSR tensor"
		H, C   = w_csr.shape
		B      = labels.size(0)

		# 1.  Build a *tiny* one-hot matrix for the batch labels (dense)
		#     shape = (B, C)  \u2192  transpose \u2192 (C, B)
		one_hot_T = F.one_hot(labels, num_classes=C).to(dtype=w_csr.dtype).t()   # (C, B)

		# 2.  Sparse-dense matmul:  (H, C) � (C, B)  \u2192  (H, B)
		#     PyTorch supports CSR � dense out-> dense.
		out_T = torch.matmul(w_csr, one_hot_T)                                   # (H, B)

		# 3.  Transpose to match the dense reference (B, H)
		return out_T.t()                                                         # (B, H)





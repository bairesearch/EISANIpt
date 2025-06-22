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
from EISANIpt_EISANImodelPrune import get_row_nnz
import torch.nn.functional as F

# -----------------------------
# Output layer
# -----------------------------

def calculateOutputLayer(self, trainOrTest, layerActivations, y):
	outputActivations = torch.zeros(batchSize, self.config.numberOfClasses, device=device)

	actLayerIndex = 0
	for layerIdSuperblock in range(recursiveSuperblocksNumber):
		for layerIdHidden in range(self.config.numberOfHiddenLayers):
			isLastLayer = (layerIdSuperblock==recursiveSuperblocksNumber-1) and (layerIdHidden==self.config.numberOfHiddenLayers-1)
			if(not useOutputConnectionsLastLayer or isLastLayer):
				act = layerActivations[actLayerIndex]
				uniqueLayerIndex = self._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden)
				
				if(not generateConnectionsAfterPropagating):
					# Training: reinforce output connections
					if trainOrTest and y is not None:
						update_output_connections(self, uniqueLayerIndex, act, y, device)
				
				weights = self.outputConnectionMatrix[uniqueLayerIndex]
					
				if(debugSequentialSANIactivationsOutputs):
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
					sparseCSRmatMulMethod = "orig"
					if(sparseCSRmatMulMethod == "orig"):
						outputActivations += act.float() @ weights 	# cast act to float for matmul
					'''
					#sparseCSRmatMulMethod = "chunkedDense"
					#sparseCSRmatMulMethod = "singleRow"
					elif(sparseCSRmatMulMethod == "chunkedDense"):
						CHUNK = 1000000   # rows per micro-matmul, tweak for your GPU
						for offset in range(0, act.size(1), CHUNK):
							r0, r1 = offset, min(offset + CHUNK, act.size(1))
							chunk  = act[:, r0:r1].float()					  # (B, chunk)
							subcsr = csr_row_block(weights, r0, r1)			# (chunk, C)
							outputActivations += chunk @ subcsr				 # (B, C)
					elif(sparseCSRmatMulMethod == "singleRow"):
						for sample in range(act.size(0)):				# small batch, e.g. 1-4
							active_rows = (act[sample] != 0).nonzero(as_tuple=True)[0]
							if active_rows.numel():
								vec = csr_row_gather_sum(active_rows, weights, self.config.numberOfClasses)
								outputActivations[sample] += vec		 # shapes now match
					'''
				else:
					outputActivations += act.float() @ weights 	# cast act to float for matmul
					
				#print("uniqueLayerIndex = ", uniqueLayerIndex)
				#print("act = ", act)	
				#print("outputActivations = ", outputActivations)
				
				if(generateConnectionsAfterPropagating):
					# Training: reinforce output connections
					if trainOrTest and y is not None:
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

	# -- 0. gather active (row, col) pairs ------------------------------
	sample_idx, neuron_idx = (activation != 0).nonzero(as_tuple=True)
	if neuron_idx.numel() == 0:
		return
		
	class_idx  = y[sample_idx]

	if(useSparseOutputMatrix):
	
		tempDevice = device	#'cpu'
		
		# ------------------------------------------------------------------
		# 1) Gather NEW (row = hidden-neuron , col = class) pairs
		# ------------------------------------------------------------------

		rows_new  = neuron_idx.to(torch.int64)
		cols_new  = class_idx.to(torch.int64)

		# Deduplicate inside the mini-batch
		edge_ids	   = rows_new * self.config.numberOfClasses + cols_new
		uniq_ids	   = torch.unique(edge_ids)
		rows_new	   = (uniq_ids // self.config.numberOfClasses)
		cols_new	   = (uniq_ids %  self.config.numberOfClasses)

		use_bool	   = useBinaryOutputConnections
		dtype_val	  = torch.bool if use_bool else torch.float32
		delta_vals_cpu = torch.ones(uniq_ids.size(0), dtype=dtype_val, device=tempDevice)

		if(tempDevice=='cpu'):
			rows_new = rows_new.cpu()
			cols_new = cols_new.cpu()
		
		# ------------------------------------------------------------------
		# 2) Build a *delta* COO tensor (CPU) with ONLY the new edges
		# ------------------------------------------------------------------
		delta_cpu = torch.sparse_coo_tensor(torch.stack([rows_new, cols_new]), delta_vals_cpu, size=self.outputConnectionMatrix[hiddenLayerIdx].shape, dtype=dtype_val, device=tempDevice).coalesce()

		# ------------------------------------------------------------------
		# 3) Merge delta with the master matrix  (all on CPU for safety)
		# ------------------------------------------------------------------
		master_csr_cpu = self.outputConnectionMatrix[hiddenLayerIdx]
		if(tempDevice=='cpu'):
			master_csr_cpu = master_csr_cpu.cpu()
		master_coo_cpu = master_csr_cpu.to_sparse_coo()

		if use_bool:
			# logical OR  ->  add ints then clamp to 0/1 and cast back to bool
			merged_int   = (master_coo_cpu.to(torch.int8) + delta_cpu.to(torch.int8)).coalesce()
			merged_vals  = (merged_int.values() > 0)
			merged_coo   = torch.sparse_coo_tensor(merged_int.indices(), merged_vals, size=master_csr_cpu.shape, dtype=torch.bool, device=tempDevice)
		else:
			# float counter  -> straight addition
			merged_coo   = (master_coo_cpu + delta_cpu).coalesce()

		# ------------------------------------------------------------------
		# 4) Back to CSR on the GPU for fast mat-mul
		# ------------------------------------------------------------------
		merged_csr_gpu = merged_coo.to_sparse_csr().to(device)
		self.outputConnectionMatrix[hiddenLayerIdx] = merged_csr_gpu

	else:
		mat = self.outputConnectionMatrix[hiddenLayerIdx]
		
		if useBinaryOutputConnections:
			new_vals = torch.ones_like(neuron_idx, dtype=torch.bool, device=device)
		else:
			new_vals = torch.ones_like(neuron_idx, dtype=torch.float32, device=device)
		if useBinaryOutputConnections:
			mat[neuron_idx, class_idx] = True						# logical OR
		else:
			mat[neuron_idx, class_idx] += 1.0						# count	

	'''
	#orig before optimisation;
	
	mat = self.outputConnectionMatrix[hiddenLayerIdx]

	# -- gather all (row = neuron , col = class) pairs in one shot ---------
	sample_idx, neuron_idx = (activation != 0).nonzero(as_tuple=True)
	if neuron_idx.numel() == 0:
		return													# nothing active

	class_idx = y[sample_idx]									# align with rows

	if useBinaryOutputConnections:
		new_vals = torch.ones_like(neuron_idx, dtype=torch.bool, device=device)
	else:
		new_vals = torch.ones_like(neuron_idx, dtype=torch.float32, device=device)

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
			mat[neuron_idx, class_idx] += 1.0						# count
	'''	

# -----------------------------------------------------------------
# Update hidden-neuron accuracy statistics (soft-max vote)
# -----------------------------------------------------------------

def updateHiddenNeuronAccuracyStatistics(self, trainOrTest, layerActivations, y):
	if(limitOutputConnections and limitOutputConnectionsBasedOnAccuracy):
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
		#     This loop is tiny: it iterates over *batch size*, not classes×rows.
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

		# 2.  Sparse-dense matmul:  (H, C) × (C, B)  \u2192  (H, B)
		#     PyTorch supports CSR × dense out-> dense.
		out_T = torch.matmul(w_csr, one_hot_T)                                   # (H, B)

		# 3.  Transpose to match the dense reference (B, H)
		return out_T.t()                                                         # (B, H)




'''
if(sparseCSRmatMulMethod == "singleRow"):
	@staticmethod
	def csr_row_gather_sum(rows_active: torch.Tensor,
								 csr: torch.Tensor,
								 num_classes: int) -> torch.Tensor:
		"""
		Sum the CSR rows in `rows_active` into a 1-D float32 vector.
		* rows_active  (K,) int64 on the same CUDA device as csr
		* csr		  sparse_csr [H, C]
		returns		(C,) float32, CUDA
		"""

		# --- CSR buffers (all CUDA) ---------------------------------------
		crow = csr.crow_indices()
		col  = csr.col_indices()
		val  = csr.values().float()				 # cast once (bool->float)

		# --- gather start / length per active row -------------------------
		start  = crow[rows_active]				  # (K,)
		end	= crow[rows_active + 1]
		lens   = end - start						# (K,)
		tot_nnz = int(lens.sum().item())
		if tot_nnz == 0:
			return torch.zeros(num_classes, device=csr.device)

		# --- build flat indices WITHOUT Python loops ----------------------
		# each row i contributes a slice [start[i], start[i]+lens[i])
		base   = start.repeat_interleave(lens)	  # (E nnz,)
		offs   = torch.arange(tot_nnz, device=csr.device) - torch.repeat_interleave(lens.cumsum(dim=0) - lens, lens)
		idx	= base + offs						# absolute positions into col/val

		cols_g = col[idx]						   # (E nnz,)
		vals_g = val[idx]

		# --- scatter-add into dense output --------------------------------
		out = torch.zeros(num_classes, device=csr.device, dtype=torch.float32)
		out.index_add_(0, cols_g, vals_g)
		return out
if(sparseCSRmatMulMethod == "chunkedDense"):
	def csr_row_block(csr: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
		"""
		Return rows [r0:r1] of a CSR tensor **without** copying col/val data.
		Works on CUDA.  (r1 is exclusive.)
		"""
		assert csr.layout == torch.sparse_csr
		crow = csr.crow_indices()
		col  = csr.col_indices()
		val  = csr.values()

		# row-pointers for the sub-block
		start_ptr = crow[r0].item()
		end_ptr   = crow[r1].item()

		crow_sub  = crow[r0:r1+1] - start_ptr		  # shift so first row starts at 0
		col_sub   = col[start_ptr:end_ptr]
		val_sub   = val[start_ptr:end_ptr]

		return torch.sparse_csr_tensor(crow_sub, col_sub, val_sub, size=(r1 - r0, csr.size(1)), dtype=csr.dtype, device=csr.device)
'''

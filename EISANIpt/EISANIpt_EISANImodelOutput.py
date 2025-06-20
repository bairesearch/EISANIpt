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
				if(debugSequentialSANIactivationsLoops):
					print("uniqueLayerIndex = ", uniqueLayerIndex)
				
				if(not generateConnectionsAfterPropagating):
					# Training: reinforce output connections
					if trainOrTest and y is not None:
						update_output_connections(self, uniqueLayerIndex, act, y, device)
				
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

				weights = normaliseOutputConnectionWeights(self, weights.float())	# cast weights to float for matmul

				if(useSparseOutputMatrix):
					#matMulMethod = "orig"
					#matMulMethod = "chunkedDense"
					matMulMethod = "singleRow"
					if(matMulMethod == "orig"):
						outputActivations += act.float() @ weights 	# cast act to float for matmul
					elif(matMulMethod == "chunkedDense"):
						CHUNK = 1_000_000   # rows per micro-matmul, tweak for your GPU
						for offset in range(0, act.size(1), CHUNK):
							r0, r1 = offset, min(offset + CHUNK, act.size(1))
							chunk  = act[:, r0:r1].float()                      # (B, chunk)
							subcsr = csr_row_block(weights, r0, r1)            # (chunk, C)
							outputActivations += chunk @ subcsr                 # (B, C)
					elif(matMulMethod == "singleRow"):
						for sample in range(act.size(0)):				# small batch, e.g. 1-4
							active_rows = (act[sample] != 0).nonzero(as_tuple=True)[0]
							if active_rows.numel():
								vec = csr_row_gather_sum(active_rows, weights, self.config.numberOfClasses)
								outputActivations[sample] += vec		 # shapes now match
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
			weights =  torch.sparse_csr_tensor(weights.crow_indices(), weights.col_indices(), vals, size=weights.shape, device=weights.device)
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
			# logical OR  \u2192  add ints then clamp to 0/1 and cast back to bool
			merged_int   = (master_coo_cpu.to(torch.int8) + delta_cpu.to(torch.int8)).coalesce()
			merged_vals  = (merged_int.values() > 0)
			merged_coo   = torch.sparse_coo_tensor(merged_int.indices(), merged_vals, size=master_csr_cpu.shape, dtype=torch.bool, device=tempDevice)
		else:
			# float counter  \u2192 straight addition
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

# -----------------------------------------------------------------
# Update hidden-neuron accuracy statistics (soft-max vote)
# -----------------------------------------------------------------

def updateHiddenNeuronAccuracyStatistics(self):
	if(limitOutputConnectionsBasedOnAccuracy):
		if y is not None:												# labels available
			#note if useOutputConnectionsLastLayer, only the last index in hiddenNeuronPredictionAccuracy will be valid

			for lidx, act in enumerate(layerActivations):

				# 1) normalise & soft-max output weights
				w_norm  = normaliseOutputConnectionWeights(self.outputConnectionMatrix[lidx])
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

@staticmethod
def csr_row_gather_sum(rows_active: torch.Tensor,
							 csr: torch.Tensor,
							 num_classes: int) -> torch.Tensor:
	"""
	Sum the CSR rows in `rows_active` into a 1-D float32 vector.
	\u2022 rows_active  (K,) int64 on the same CUDA device as csr
	\u2022 csr		  sparse_csr [H, C]
	returns		(C,) float32, CUDA
	"""

	# --- CSR buffers (all CUDA) ---------------------------------------
	crow = csr.crow_indices()
	col  = csr.col_indices()
	val  = csr.values().float()				 # cast once (bool\u2192float)

	# --- gather start / length per active row -------------------------
	start  = crow[rows_active]				  # (K,)
	end	= crow[rows_active + 1]
	lens   = end - start						# (K,)
	tot_nnz = int(lens.sum().item())
	if tot_nnz == 0:
		return torch.zeros(num_classes, device=csr.device)

	# --- build flat indices WITHOUT Python loops ----------------------
	# each row i contributes a slice [start[i], start[i]+lens[i])
	base   = start.repeat_interleave(lens)	  # (\u03a3nnz,)
	offs   = torch.arange(tot_nnz, device=csr.device) - torch.repeat_interleave(lens.cumsum(dim=0) - lens, lens)
	idx	= base + offs						# absolute positions into col/val

	cols_g = col[idx]						   # (\u03a3nnz,)
	vals_g = val[idx]

	# --- scatter-add into dense output --------------------------------
	out = torch.zeros(num_classes, device=csr.device, dtype=torch.float32)
	out.index_add_(0, cols_g, vals_g)
	return out



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

	crow_sub  = crow[r0:r1+1] - start_ptr          # shift so first row starts at 0
	col_sub   = col[start_ptr:end_ptr]
	val_sub   = val[start_ptr:end_ptr]

	return torch.sparse_csr_tensor(crow_sub, col_sub, val_sub, size=(r1 - r0, csr.size(1)), dtype=csr.dtype, device=csr.device)

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

def update_output_connections(self, hiddenLayerIdx: int, activation: torch.Tensor, y: torch.Tensor, device: torch.device,) -> None:
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



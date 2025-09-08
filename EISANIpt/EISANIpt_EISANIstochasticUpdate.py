"""EISANIpt_EISANIstochasticUpdate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt stochastic update

Based on LREANNpt_SUANN.py

"""

import torch as pt
from ANNpt_globalDefs import *
import EISANIpt_EISANImodel
import EISANIpt_EISANImodelContinuousVarEncoding
if useSequentialSANI:
	import EISANIpt_EISANImodelSequential as _modelSeq
else:
	import EISANIpt_EISANImodelSummation as _modelSum

# -------------------------------------------------------------
# Stochastic update wrapper (trial-and-keep)
# -------------------------------------------------------------

def performStochasticUpdate(model, trainOrTest, x, y, optim=None, l=None, batchIndex=None, fieldTypeList=None):
	"""
	use stochastic trial updates instead of default growth.
	- When useStochasticUpdates is False: delegates to model(...)
	- When True and train phase: proposes small independent connection changes
	  and keeps only those that improve loss (1-accuracy) on the current batch.
	Notes:
	  - Evaluations are performed with trainOrTest=False to freeze default updates.
	  - A final forward pass with trainOrTest=True applies normal output updates.
	"""

	with pt.no_grad():
	
		if not debugStochasticUpdatesDisableGrowth:
			# Baseline: use a more sensitive scoring function than discrete accuracy
			# so that small random changes in hidden connections can be detected
			# even when predictions (argmax) do not flip.
			best_loss = _score_batch_cross_entropy(model, x, y)

			trials = max(1, int(stochasticUpdatesPerBatch))
			for trialIndex in range(trials):
				# Specification (updated): always propose a RANDOM change.
				# - Static: flip a single random existing connection.
				# - Dynamic: choose a random (i,j); if absent add a single new connection with random sign,
				#            else flip the sign of the existing connection.
				change = _propose_change(model)
				if change is None:
					continue

				_apply_change(model, change)
				# Score with outputs frozen; no model() call to avoid any side-effects
				# such as optional stats updates. Evaluate CE over current logits.
				new_loss = _score_batch_cross_entropy(model, x, y)

				if(debugStochasticUpdates):
					print("trialIndex = ", trialIndex)
					print("proposed change = ", change)
					print("new_loss = ", new_loss)
					print("best_loss = ", best_loss)

				if new_loss < best_loss:
					# keep the change, update baseline
					best_loss = new_loss
					_finalise_change(model, change, keep=True)
				else:
					# revert
					_finalise_change(model, change, keep=False)

		# Dense output-layer backprop: fully-connected hidden->output when useStochasticUpdates
		# Compute hidden activations once, update dense output weights with a single-layer SGD step,
		# and report accuracy/loss based on current logits.
		initActivation = EISANIpt_EISANImodelContinuousVarEncoding.continuousVarEncoding(model, x)
		# Get hidden activations without triggering any dynamic growth
		if useSequentialSANI:
			layerActivations = _modelSeq.sequentialSANIpassHiddenLayers(model, False, batchIndex, 0, initActivation)
		else:
			layerActivations = _modelSum.summationSANIpassHiddenLayers(model, False, initActivation)

		# Build logits and collect per-layer activations for gradient update
		C = int(model.config.numberOfClasses)
		B = int(x.size(0))
		logits = pt.zeros((B, C), device=x.device, dtype=pt.float32)
		per_layer_acts = []
		per_layer_idx  = []
		actLayerIndex = 0
		for layerIdSuperblock in range(recursiveSuperblocksNumber):
			for layerIdHidden in range(model.config.numberOfHiddenLayers):
				isLastLayer = (layerIdSuperblock==recursiveSuperblocksNumber-1) and (layerIdHidden==model.config.numberOfHiddenLayers-1)
				if (not useOutputConnectionsLastLayer) or isLastLayer:
					act = layerActivations[actLayerIndex].float()
					uniqueLayerIndex = model._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden)
					W = model.outputConnectionMatrix[uniqueLayerIndex].float()
					logits = logits + act @ W
					per_layer_acts.append(act)
					per_layer_idx.append(uniqueLayerIndex)
				actLayerIndex += 1

		pred = pt.argmax(logits, dim=1)
		correct = (pred == y).sum().item()
		accuracy = correct / max(1, y.size(0))
		# Single-layer backprop gradient step (no autograd through hidden)
		probs = pt.softmax(logits, dim=1)
		y_onehot = pt.nn.functional.one_hot(y, num_classes=C).to(probs.dtype)
		delta = (probs - y_onehot) / max(1, B)
		lr = stochasticOutputLearningRate
		for act, lidx in zip(per_layer_acts, per_layer_idx):
			grad = act.t() @ delta
			model.outputConnectionMatrix[lidx] = model.outputConnectionMatrix[lidx].float().add_( -lr * grad )

		loss = EISANIpt_EISANImodel.Loss(1.0 - float(accuracy))
		accuracy = float(accuracy)
		return loss, accuracy

def _score_batch_cross_entropy(model, x, y) -> float:
	"""Compute a smooth loss (mean cross-entropy) from current hidden weights.

	This uses the same hidden-pass and dense output aggregation as the
	post-trial update step, but without mutating model state. It is more
	sensitive than accuracy-based scoring, allowing random proposals to
	produce measurable loss differences even when argmax is unchanged.
	"""
	# Encode inputs and obtain hidden activations without any dynamic growth
	initActivation = EISANIpt_EISANImodelContinuousVarEncoding.continuousVarEncoding(model, x)
	if initActivation is None:
		return 1.0
	if useSequentialSANI:
		layerActivations = _modelSeq.sequentialSANIpassHiddenLayers(model, False, None, 0, initActivation)
	else:
		layerActivations = _modelSum.summationSANIpassHiddenLayers(model, False, initActivation)

	# Aggregate logits over configured hidden layers
	C = int(model.config.numberOfClasses)
	B = int(x.size(0))
	logits = pt.zeros((B, C), device=x.device, dtype=pt.float32)
	actLayerIndex = 0
	for layerIdSuperblock in range(recursiveSuperblocksNumber):
		for layerIdHidden in range(model.config.numberOfHiddenLayers):
			isLastLayer = (layerIdSuperblock==recursiveSuperblocksNumber-1) and (layerIdHidden==model.config.numberOfHiddenLayers-1)
			if (not useOutputConnectionsLastLayer) or isLastLayer:
				act = layerActivations[actLayerIndex].float()
				uniqueLayerIndex = model._getUniqueLayerIndex(layerIdSuperblock, layerIdHidden)
				W = model.outputConnectionMatrix[uniqueLayerIndex].float()
				logits = logits + act @ W
			actLayerIndex += 1

	# Cross-entropy (no autograd): -mean log p(y|x)
	log_probs = pt.log_softmax(logits, dim=1)
	# Guard against invalid labels (e.g., padding) by clamping
	y_clamped = y.clamp(min=0, max=C-1)
	ce = -log_probs[pt.arange(B, device=x.device), y_clamped].mean()
	return float(ce.item())


def rand_index(t: pt.Tensor) -> int:
	"""Return a random flat index for tensor `t`."""
	numel = t.numel()
	return pt.randint(numel, (), device=t.device).item()


# -------------------------------------------------------------
# Optional compatibility helpers (from SUANN-style interface)
# -------------------------------------------------------------

def pertubation_function(x):  # kept for API parity; unused for binary sparse
	return 1


def perturb_once(model, name: str = '', p=None, idx: int = 0):
	"""
	Select a candidate change on the model (hidden connections).
	Returns (change_handle, backup) where `change_handle` carries coordinates.
	"""
	change = _propose_change(model)
	if change is None:
		return None, None
	_apply_change(model, change)
	# In this binary/sparse context backup info is embedded in `change` itself
	return change, change


def update_weight(model, idx: int, backup, name: str, p, e, loss1, loss2):
	"""
	Keep change if loss2 < loss1, else revert.
	`backup` is the `change` dict returned by perturb_once.
	"""
	keep = loss2 < loss1
	_finalise_change(model, backup, keep=keep)


# -------------------------------------------------------------
# Internal helpers for stochastic proposals
# -------------------------------------------------------------

def _get_prev_layer_size(model, hiddenLayerIdx: int) -> int:
	"""Return the input size feeding into `hiddenLayerIdx`.

	This inspects the shape of the existing hidden connection matrix so it
	works for any layer index (eg. with recursive or dynamically generated
	layers) rather than assuming only the first layer differs in size.
	"""
	mat = model.hiddenConnectionMatrix[hiddenLayerIdx][0]
	return int(mat.shape[1])

def _choose_random_layer_and_segment(model):
	"""Select a random hidden layer and segment for a proposal.

	By default layers are selected uniformly from the range
	``[0, model.numberUniqueHiddenLayers)``.  If a global variable
	``stochasticLayerBias`` (>0) is defined, earlier layers are favoured by
	weighting the sampling probability as ``1/(i+1)**stochasticLayerBias``.
	"""
	if stochasticLayerBias and stochasticLayerBias > 0:
		weights = pt.tensor([1.0 / ((i + 1) ** stochasticLayerBias) for i in range(model.numberUniqueHiddenLayers)], dtype=pt.float32)
		probs = weights / weights.sum()
		hiddenLayerIdx = int(pt.multinomial(probs, 1).item())
	else:
		hiddenLayerIdx = int(pt.randint(model.numberUniqueHiddenLayers, (1,), device=pt.device('cpu')).item())
	seg = int(pt.randint(numberOfSegmentsPerNeuron, (1,), device=pt.device('cpu')).item())
	return hiddenLayerIdx, seg


def _propose_change(model):
	"""
	Create a small candidate change on hidden connections.
	- If useDynamicGeneratedHiddenConnections False: flip sign of an existing connection.
	- If True: add a new single connection (or flip sign if it already exists with opposite sign).
	Returns a dict describing the change or None if not applicable.
	"""
	# Only implemented for non-EI hidden connections
	if useEIneurons:
		return None

	hidx, seg = _choose_random_layer_and_segment(model)

	# Retrieve the weight matrix for this hidden layer & segment
	mat = model.hiddenConnectionMatrix[hidx][seg]
	prevSize = _get_prev_layer_size(model, hidx)
	layerSize = int(model.config.hiddenLayerSize)

	if not useDynamicGeneratedHiddenConnections:
		# Flip the sign of one existing connection
		if useSparseHiddenMatrix:
			mat = mat.coalesce()
			idx = mat.indices()
			vals = mat.values()
			nnz = vals.numel()
			if nnz == 0:
				return None
			k = int(pt.randint(nnz, (1,), device=vals.device).item())
			i = int(idx[0, k].item())
			j = int(idx[1, k].item())
			old_val = bool(vals[k].item())
			change = {
				'type': 'flip_existing_sparse',
				'hidx': hidx,
				'seg': seg,
				'i': i,
				'j': j,
				'old': old_val,
			}
			return change
		else:
			# dense int8 matrix: pick a row with any non-zero, then a column
			# try a few times to find a non-empty row
			for _ in range(5):
				i = int(pt.randint(layerSize, (1,), device=mat.device).item())
				row = mat[i]
				nz = (row != 0).nonzero(as_tuple=False)
				if nz.numel() > 0:
					j = int(nz[pt.randint(nz.size(0), (1,), device=mat.device).item(), 0].item())
					old_val = int(mat[i, j].item())
					change = {
						'type': 'flip_existing_dense',
						'hidx': hidx,
						'seg': seg,
						'i': i,
						'j': j,
						'old': old_val,
					}
					return change
			return None
	else:
		# Dynamic: propose single random addition or flip
		i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
		j = int(pt.randint(prevSize,  (1,), device=pt.device('cpu')).item())
		if useSparseHiddenMatrix:
			co = mat.coalesce()
			idx = co.indices()
			vals = co.values()
			mask_ij = (idx[0] == i) & (idx[1] == j)
			if mask_ij.any():
				# flip existing single connection
				kpos = int(mask_ij.nonzero(as_tuple=False)[0, 0].item())
				old_val = bool(vals[kpos].item())
				return {
					'type': 'flip_existing_sparse',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'old': old_val,
				}
			else:
				# add a single random connection
				val_bool = bool(int(pt.randint(2, (1,), device=pt.device('cpu')).item()))
				return {
					'type': 'add_new_sparse',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'val': val_bool,
				}
		else:
			old = int(mat[i, j].item())
			if old == 0:
				# add new with random sign (+1/-1)
				val = 1 if int(pt.randint(2, (1,), device=pt.device('cpu')).item()) == 1 else -1
				return {
					'type': 'add_new_dense',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'val': val,
					'old': 0,
				}
			else:
				return {
					'type': 'flip_existing_dense',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'old': old,
				}

def _apply_change(model, change):
	hidx = change['hidx']
	seg = change['seg']
	mat = model.hiddenConnectionMatrix[hidx][seg]

	if change['type'] == 'flip_existing_sparse':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		# find the exact index again (small nnz expected per trial)
		mask = (idx[0] == change['i']) & (idx[1] == change['j'])
		if mask.any():
			k = int(mask.nonzero(as_tuple=False)[0, 0].item())
			vals[k] = ~vals[k]
			model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(idx, vals, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'add_new_sparse':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		new_idx = pt.tensor([[change['i']], [change['j']]], dtype=idx.dtype, device=idx.device)
		new_val = pt.tensor([change['val']], dtype=vals.dtype, device=vals.device)
		all_idx = pt.cat([idx, new_idx], dim=1)
		all_val = pt.cat([vals, new_val], dim=0)
		model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(all_idx, all_val, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'flip_existing_dense':
		mat[change['i'], change['j']] = -mat[change['i'], change['j']]
	elif change['type'] == 'add_new_dense':
		mat[change['i'], change['j']] = pt.tensor(change['val'], dtype=mat.dtype, device=mat.device)


def _finalise_change(model, change, keep: bool):
	if keep:
		return  # already applied

	# revert
	hidx = change['hidx']
	seg = change['seg']
	mat = model.hiddenConnectionMatrix[hidx][seg]

	if change['type'] == 'flip_existing_sparse':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		mask = (idx[0] == change['i']) & (idx[1] == change['j'])
		if mask.any():
			k = int(mask.nonzero(as_tuple=False)[0, 0].item())
			# restore
			vals[k] = pt.tensor(change['old'], dtype=vals.dtype, device=vals.device)
			model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(idx, vals, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'add_new_sparse':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		# remove the (i,j) we just added
		mask = ~((idx[0] == change['i']) & (idx[1] == change['j']))
		new_idx = idx[:, mask]
		new_vals = vals[mask]
		model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(new_idx, new_vals, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'flip_existing_dense':
		mat[change['i'], change['j']] = pt.tensor(change['old'], dtype=mat.dtype, device=mat.device)
	elif change['type'] == 'add_new_dense':
		mat[change['i'], change['j']] = pt.tensor(change['old'], dtype=mat.dtype, device=mat.device)


def _clone_tensor_like(t: pt.Tensor):
	"""Clone dense/CSR/COO tensors robustly."""
	if t is None:
		return None
	try:
		return t.clone()
	except Exception:
		# Fallbacks for older torch versions
		if t.is_sparse_csr:
			return pt.sparse_csr_tensor(t.crow_indices().clone(), t.col_indices().clone(), t.values().clone(), size=t.shape, device=t.device, dtype=t.dtype)
		elif t.is_sparse:
			co = t.coalesce()
			return pt.sparse_coo_tensor(co.indices().clone(), co.values().clone(), size=co.shape, device=co.device, dtype=co.dtype)
		else:
			return t.detach().clone()


def _snapshot_output_state(model):
	"""Snapshot mutable output-related state so trials can run side-effect free."""
	snap = {}
	if hasattr(model, 'outputConnectionMatrix') and isinstance(model.outputConnectionMatrix, list):
		snap['outputConnectionMatrix'] = [_clone_tensor_like(w) for w in model.outputConnectionMatrix]
	# Optional stats that may update during forward when pruning is enabled
	if hasattr(model, 'hiddenNeuronPredictionAccuracy'):
		snap['hiddenNeuronPredictionAccuracy'] = [t.clone() for t in model.hiddenNeuronPredictionAccuracy]
	if hasattr(model, 'hiddenNeuronUsage'):
		snap['hiddenNeuronUsage'] = [t.clone() for t in model.hiddenNeuronUsage]
	return snap


def _restore_output_state(model, snap):
	"""Restore state captured by _snapshot_output_state."""
	if 'outputConnectionMatrix' in snap:
		for i, w in enumerate(snap['outputConnectionMatrix']):
			model.outputConnectionMatrix[i] = w
	if 'hiddenNeuronPredictionAccuracy' in snap:
		for i, t in enumerate(snap['hiddenNeuronPredictionAccuracy']):
			model.hiddenNeuronPredictionAccuracy[i] = t
	if 'hiddenNeuronUsage' in snap:
		for i, t in enumerate(snap['hiddenNeuronUsage']):
			model.hiddenNeuronUsage[i] = t

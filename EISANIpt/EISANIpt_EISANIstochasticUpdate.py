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

from ANNpt_globalDefs import *
import EISANIpt_EISANImodel
import EISANIpt_EISANImodelContinuousVarEncoding
import torch as pt


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
		# Baseline: evaluate loss with outputs FROZEN (no train step),
		# and snapshot/restore to avoid mutating any auxiliary stats.
		snap = _snapshot_output_state(model)
		loss_obj, acc = model(False, x, y, optim, l, batchIndex, fieldTypeList)
		best_loss = float(loss_obj.item())
		_restore_output_state(model, snap)

		trials = max(1, int(stochasticUpdatesPerBatch))
		for trialIndex in range(trials):
			print("trialIndex = ", trialIndex)
			# Prefer guided proposals using current batch activations; fallback to random
			change = _propose_change_guided(model, x)
			if change is None:
				change = _propose_change(model)
			print("proposed change = ", change)
			if change is None:
				continue

			_apply_change(model, change)
			# Score with outputs frozen; snapshot/restore to keep side-effect free
			snap_trial = _snapshot_output_state(model)
			new_loss_obj, _ = model(False, x, y, optim, l, batchIndex, fieldTypeList)
			new_loss = float(new_loss_obj.item())
			_restore_output_state(model, snap_trial)
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
		if initActivation is None:
			return EISANIpt_EISANImodel.Loss(1.0), 0.0
		# Get hidden activations without triggering any dynamic growth
		if 'useSequentialSANI' in globals() and useSequentialSANI:
			import EISANIpt_EISANImodelSequential as _modelSeq
			layerActivations = _modelSeq.sequentialSANIpassHiddenLayers(model, False, batchIndex, 0, initActivation)
		else:
			import EISANIpt_EISANImodelSummation as _modelSum
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
		lr = float(stochasticOutputLearningRate) if 'stochasticOutputLearningRate' in globals() else 0.05
		for act, lidx in zip(per_layer_acts, per_layer_idx):
			grad = act.t() @ delta
			model.outputConnectionMatrix[lidx] = model.outputConnectionMatrix[lidx].float().add_( -lr * grad )

		return EISANIpt_EISANImodel.Loss(1.0 - float(accuracy)), float(accuracy)


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
	# For non-sequential summation model: first hidden layer connects from encodedFeatureSize
	if hiddenLayerIdx == 0:
		return int(model.encodedFeatureSize)
	else:
		return int(model.config.hiddenLayerSize)


def _choose_random_layer_and_segment(model):
	# Focus early growth on the first hidden layer so new inputs can
	# actually propagate. This avoids proposing changes on deeper layers
	# that have no upstream activation yet.
	hiddenLayerIdx = 0
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
		# Propose adding a bundle of new positive connections for a single neuron
		# in order to cross activation threshold in one trial.
		i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
		k = int(model.config.numberOfSynapsesPerSegment)
		if useSparseHiddenMatrix:
			co = mat.coalesce()
			idx = co.indices()
			vals = co.values()
			# Get existing columns for row i to avoid duplicates
			if idx.numel() > 0:
				row_mask = (idx[0] == i)
				exist_cols = idx[1, row_mask] if row_mask.any() else None
			else:
				exist_cols = None
			J = []
			tries = 0
			max_tries = max(10*k, 50)
			while len(J) < k and tries < max_tries:
				j = int(pt.randint(prevSize, (1,), device=pt.device('cpu')).item())
				if exist_cols is not None and (exist_cols == j).any():
					tries += 1
					continue
				if j in J:
					tries += 1
					continue
				J.append(j)
				tries += 1
			if len(J) == 0:
				return None
			return {
				'type': 'add_new_sparse_bundle',
				'hidx': hidx,
				'seg': seg,
				'i': i,
				'J': J,
				'val': True,  # add +1 connections to encourage activation
			}
		else:
			J = []
			old_vals = []
			tries = 0
			while len(J) < k and tries < 10*k:
				j = int(pt.randint(prevSize, (1,), device=pt.device('cpu')).item())
				if j in J:
					tries += 1
					continue
				old = int(mat[i, j].item())
				if old != 1:
					J.append(j)
					old_vals.append(old)
				tries += 1
			if len(J) == 0:
				return None
			return {
				'type': 'add_new_dense_bundle',
				'hidx': hidx,
				'seg': seg,
				'i': i,
				'J': J,
				'old_list': old_vals,
				'val': 1,
			}


def _propose_change_guided(model, x):
	"""Propose a change that targets currently-active presynaptic inputs in layer 0.

	For dynamic sparse matrices, add k new +1 synapses from columns active in this batch
	to a single neuron in the first hidden layer. This increases the chance that a segment
	will cross its activation threshold in the same batch, producing measurable loss deltas.
	"""
	if useEIneurons:
		return None
	if not useDynamicGeneratedHiddenConnections:
		# Static case: flip an existing connection aligned with currently-active inputs.
		# Focus on the first hidden layer, segment 0
		hidx, seg = 0, 0
		mat = model.hiddenConnectionMatrix[hidx][seg]
		layerSize = int(model.config.hiddenLayerSize)

		# Compute current input activations to guide selection
		initActivation = EISANIpt_EISANImodelContinuousVarEncoding.continuousVarEncoding(model, x)
		if initActivation is None:
			return None
		B = initActivation.size(0)
		if B == 0:
			return None
		b = int(pt.randint(B, (1,), device=pt.device('cpu')).item())
		active_cols_b = (initActivation[b] != 0).nonzero(as_tuple=False).squeeze(1)
		if active_cols_b.numel() == 0:
			return None

		if useSparseHiddenMatrix:
			co = mat.coalesce()
			idx = co.indices()
			vals = co.values()
			if idx.numel() == 0:
				return None
			# try a few rows to find an active-aligned synapse
			for _ in range(5):
				i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
				mask_row = (idx[0] == i)
				if not mask_row.any():
					continue
				row_positions = mask_row.nonzero(as_tuple=False).squeeze(1)
				cols_i = idx[1, mask_row]
				vals_i = vals[mask_row]
				# candidates where presynaptic j is active in sample b
				active_mask = pt.isin(cols_i, active_cols_b)
				if not active_mask.any():
					continue
				# Prefer flipping a negative weight (-1 => False) to positive
				neg_mask = active_mask & (~vals_i)
				if neg_mask.any():
					pick_rel = int(neg_mask.nonzero(as_tuple=False)[0, 0].item())
				else:
					pick_rel = int(active_mask.nonzero(as_tuple=False)[0, 0].item())
				kpos = int(row_positions[pick_rel].item())
				j = int(idx[1, kpos].item())
				old_val = bool(vals[kpos].item())
				return {
					'type': 'flip_existing_sparse',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'old': old_val,
				}
			return None
		else:
			# Dense case
			for _ in range(5):
				i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
				row = mat[i]
				nz = (row != 0).nonzero(as_tuple=False).squeeze(1)
				if nz.numel() == 0:
					continue
				# align with active presynaptic cols
				mask_active = pt.isin(nz, active_cols_b)
				nz_active = nz[mask_active]
				if nz_active.numel() == 0:
					continue
				vals_active = row[nz_active]
				neg_mask = (vals_active < 0)
				if neg_mask.any():
					pick_rel = int(neg_mask.nonzero(as_tuple=False)[0, 0].item())
				else:
					pick_rel = 0
				j = int(nz_active[pick_rel].item())
				old_val = int(row[j].item())
				return {
					'type': 'flip_existing_dense',
					'hidx': hidx,
					'seg': seg,
					'i': i,
					'j': j,
					'old': old_val,
				}
			return None

	# Only guide for the first hidden layer, segment 0
	hidx, seg = 0, 0
	mat = model.hiddenConnectionMatrix[hidx][seg]
	prevSize = _get_prev_layer_size(model, hidx)
	layerSize = int(model.config.hiddenLayerSize)
	k = int(model.config.numberOfSynapsesPerSegment)

	# Compute initial activation (prevActivation for layer 0)
	initActivation = EISANIpt_EISANImodelContinuousVarEncoding.continuousVarEncoding(model, x)
	if initActivation is None:
		return None
	# Target a single sample so we ensure the chosen synapses are on for that sample
	B = initActivation.size(0)
	b = int(pt.randint(B, (1,), device=pt.device('cpu')).item())
	active_cols_b = (initActivation[b] != 0).nonzero(as_tuple=False).squeeze(1)
	if active_cols_b.numel() == 0:
		return None

	# Pick a target neuron
	i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
	J = []
	if useSparseHiddenMatrix:
		co = mat.coalesce()
		idx = co.indices()
		# existing columns for row i
		if idx.numel() > 0:
			row_mask = (idx[0] == i)
			exist_cols = idx[1, row_mask] if row_mask.any() else None
		else:
			exist_cols = None
		# choose up to k columns from active set for sample b, avoiding duplicates
		cand = active_cols_b.tolist()
		# shuffle candidates to avoid bias
		if len(cand) > 1:
			perm = pt.randperm(len(cand), device=pt.device('cpu')).tolist()
			cand = [cand[p] for p in perm]
		for j in cand:
			if len(J) >= k:
				break
			if exist_cols is not None and (exist_cols == j).any():
				continue
			J.append(int(j))
		if len(J) == 0:
			return None
		return {
			'type': 'add_new_sparse_bundle',
			'hidx': hidx,
			'seg': seg,
			'i': i,
			'J': J,
			'val': True,
		}
	else:
		# dense case: avoid columns already at +1
		cand = active_cols_b.tolist()
		if len(cand) > 1:
			perm = pt.randperm(len(cand), device=pt.device('cpu')).tolist()
			cand = [cand[p] for p in perm]
		old_vals = []
		for j in cand:
			if len(J) >= k:
				break
			old = int(mat[i, j].item())
			if old == 1:
				continue
			J.append(int(j))
			old_vals.append(old)
		if len(J) == 0:
			return None
		return {
			'type': 'add_new_dense_bundle',
			'hidx': hidx,
			'seg': seg,
			'i': i,
			'J': J,
			'old_list': old_vals,
			'val': 1,
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
	elif change['type'] == 'add_new_sparse_bundle':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		J = change['J']
		new_idx = pt.tensor([[change['i']] * len(J), J], dtype=idx.dtype, device=idx.device)
		new_val = pt.ones((len(J),), dtype=vals.dtype, device=vals.device) if change['val'] else pt.zeros((len(J),), dtype=vals.dtype, device=vals.device)
		all_idx = pt.cat([idx, new_idx], dim=1)
		all_val = pt.cat([vals, new_val], dim=0)
		model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(all_idx, all_val, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'flip_existing_dense':
		mat[change['i'], change['j']] = -mat[change['i'], change['j']]
	elif change['type'] == 'add_new_dense':
		mat[change['i'], change['j']] = pt.tensor(change['val'], dtype=mat.dtype, device=mat.device)
	elif change['type'] == 'add_new_dense_bundle':
		for j in change['J']:
			mat[change['i'], j] = pt.tensor(change['val'], dtype=mat.dtype, device=mat.device)


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
	elif change['type'] == 'add_new_sparse_bundle':
		co = mat.coalesce()
		idx = co.indices()
		vals = co.values()
		Jten = pt.tensor(change['J'], dtype=idx.dtype, device=idx.device)
		mask = ~((idx[0] == change['i']) & pt.isin(idx[1], Jten))
		new_idx = idx[:, mask]
		new_vals = vals[mask]
		model.hiddenConnectionMatrix[hidx][seg] = pt.sparse_coo_tensor(new_idx, new_vals, size=co.shape, device=co.device, dtype=co.dtype).coalesce()
	elif change['type'] == 'flip_existing_dense':
		mat[change['i'], change['j']] = pt.tensor(change['old'], dtype=mat.dtype, device=mat.device)
	elif change['type'] == 'add_new_dense':
		mat[change['i'], change['j']] = pt.tensor(change['old'], dtype=mat.dtype, device=mat.device)
	elif change['type'] == 'add_new_dense_bundle':
		for j, old in zip(change['J'], change['old_list']):
			mat[change['i'], j] = pt.tensor(old, dtype=mat.dtype, device=mat.device)


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

"""EISANIpt_EISANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt excitatory inhibitory (EI) sequentially/summation activated neuronal input (SANI) network

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import EISANIpt_EISANImodel
import ANNpt_data
import torch as pt

from EISANIpt_EISANI_globalDefs import (
    useDynamicGeneratedHiddenConnections,
    useSparseHiddenMatrix,
    useEIneurons,
    numberOfSegmentsPerNeuron,
    useStochasticUpdates,
    stochasticUpdatesPerBatch,
)


def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=False)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)	#note "numberOfFeatures" is the raw continuous var input (without x-bit encoding)	#not used
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)
	hiddenLayerSizeSANI = EISANIpt_EISANImodel.generateHiddenLayerSizeSANI(datasetSize, trainNumberOfEpochs, numberOfLayers, numberOfConvlayers)

	if(printEISANImodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t useTabularDataset = ", useTabularDataset)
		print("\t useImageDataset = ", useImageDataset)
		print("\t useNLPDataset = ", useNLPDataset)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetSize = ", datasetSize)
		print("\t datasetEqualiseClassSamples = ", datasetEqualiseClassSamples)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t numberOfConvlayers = ", numberOfConvlayers)
		print("\t hiddenLayerSizeSANI = ", hiddenLayerSizeSANI)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t numberOfSynapsesPerSegment = ", numberOfSynapsesPerSegment)
		print("\t numberOfSegmentsPerNeuron = ", numberOfSegmentsPerNeuron)
		print("\t ---")
		print("\t useBinaryOutputConnections = ", useBinaryOutputConnections)
		print("\t useDynamicGeneratedHiddenConnections = ", useDynamicGeneratedHiddenConnections)
		print("\t useEIneurons = ", useEIneurons)
		print("\t EISANITABcontinuousVarEncodingNumBits = ", EISANITABcontinuousVarEncodingNumBits)
		print("\t numberNeuronSegmentsGeneratedPerSample = ", numberNeuronSegmentsGeneratedPerSample)
		print("\t recursiveLayers = ", recursiveLayers)
		print("\t recursiveSuperblocksNumber = ", recursiveSuperblocksNumber)
		print("\t useOutputConnectionsNormalised = ", useOutputConnectionsNormalised)
		if(limitConnections and limitOutputConnections):
			print("\t limitOutputConnectionsBasedOnPrevalence = ", limitOutputConnectionsBasedOnPrevalence)
			print("\t limitOutputConnectionsBasedOnAccuracy = ", limitOutputConnectionsBasedOnAccuracy)
		if(useNLPDataset):
			print("\t ---")
			print("\t useNLPDataset:")
			print("\t\t evalOnlyUsingTimeInvariance = ", evalOnlyUsingTimeInvariance)
			print("\t\t evalStillTrainOutputConnections = ", evalStillTrainOutputConnections)
			print("\t\t sequentialSANItimeInvariance = ", sequentialSANItimeInvariance)
			print("\t\t useSequentialSANIactivationStrength = ", useSequentialSANIactivationStrength)
			print("\t\t sequentialSANIoverlappingSegments = ", sequentialSANIoverlappingSegments)
			print("\t\t datasetTrainRows = ", datasetTrainRows)
			print("\t\t datasetTestRows = ", datasetTestRows)
		print("\t ---") 
		print("\t useStochasticUpdates = ", useStochasticUpdates)
		
		
	config = EISANIpt_EISANImodel.EISANIconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSizeSANI,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		numberOfSynapsesPerSegment = numberOfSynapsesPerSegment,
		fieldTypeList = fieldTypeList,
	)
	model = EISANIpt_EISANImodel.EISANImodel(config)
	
	print(model)

	return model


# -------------------------------------------------------------
# Stochastic update wrapper (trial-and-keep)
# -------------------------------------------------------------

def trainOrTestModel(model, trainOrTest, x, y, optim=None, l=None, batchIndex=None, fieldTypeList=None):
    """
    Wrapper to optionally use stochastic trial updates instead of default growth.
    - When useStochasticUpdates is False: delegates to model(...)
    - When True and train phase: proposes small independent connection changes
      and keeps only those that improve loss (1-accuracy) on the current batch.
    Notes:
      - Evaluations are performed with trainOrTest=False to freeze default updates.
      - A final forward pass with trainOrTest=True applies normal output updates.
    """
 
    if not (trainOrTest and useStochasticUpdates):
        return model(trainOrTest, x, y, optim, l, batchIndex, fieldTypeList)
    else:
        with pt.no_grad():
            # Baseline (eval mode to avoid modifying output weights during trials)
            loss_obj, acc = model(False, x, y, optim, l, batchIndex, fieldTypeList)
            best_loss = float(loss_obj.item())

            trials = max(1, int(stochasticUpdatesPerBatch))
            for _ in range(trials):
                change = _propose_change(model)
                if change is None:
                    continue

                _apply_change(model, change)
                new_loss_obj, _ = model(False, x, y, optim, l, batchIndex, fieldTypeList)
                new_loss = float(new_loss_obj.item())

                if new_loss < best_loss:
                    # keep the change, update baseline
                    best_loss = new_loss
                    _finalise_change(model, change, keep=True)
                else:
                    # revert
                    _finalise_change(model, change, keep=False)

            # Final training pass to update any train-time stats/output connections
            final_loss, final_acc = model(True, x, y, optim, l, batchIndex, fieldTypeList)
            return final_loss, final_acc


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
        # Propose adding a new connection (or flipping if opposite sign improves)
        i = int(pt.randint(layerSize, (1,), device=pt.device('cpu')).item())
        j = int(pt.randint(prevSize, (1,), device=pt.device('cpu')).item())
        # propose sign: True=+1, False=-1
        sign_bool = bool(pt.randint(2, (1,), device=pt.device('cpu')).item())

        if useSparseHiddenMatrix:
            mat = mat.coalesce()
            idx = mat.indices()
            vals = mat.values()
            # locate (i,j)
            if idx.numel() > 0:
                mask = (idx[0] == i) & (idx[1] == j)
                if mask.any():
                    k = int(mask.nonzero(as_tuple=False)[0, 0].item())
                    old_val = bool(vals[k].item())
                    if old_val != sign_bool:
                        return {
                            'type': 'flip_existing_sparse',
                            'hidx': hidx,
                            'seg': seg,
                            'i': i,
                            'j': j,
                            'old': old_val,
                        }
                    else:
                        return None  # same sign -> no change
            # else: add new
            return {
                'type': 'add_new_sparse',
                'hidx': hidx,
                'seg': seg,
                'i': i,
                'j': j,
                'val': sign_bool,
            }
        else:
            old = int(mat[i, j].item())
            new_val = 1 if sign_bool else -1
            if old == 0:
                return {
                    'type': 'add_new_dense',
                    'hidx': hidx,
                    'seg': seg,
                    'i': i,
                    'j': j,
                    'old': old,
                    'val': new_val,
                }
            elif old != new_val:
                return {
                    'type': 'flip_existing_dense',
                    'hidx': hidx,
                    'seg': seg,
                    'i': i,
                    'j': j,
                    'old': old,
                }
            else:
                return None


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

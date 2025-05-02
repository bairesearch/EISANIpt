"""columnarEInetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision
pip install networkx
pip install matplotlib

# Usage:
source activate pytorchsenv
python columnarEInetwork.py --epochs 1 --train-samples 500 --use-sparse --viz-neuron --viz-network --viz-full

Flags ``--viz-neuron`` / ``--viz-network`` / ``--viz-full`` correspond to options (a)/(b)/(c).  
See ``python columnarEInetwork.py --help`` for the full CLI.

# Description:
Columnar excitatory/inhibitory (EI) neuron network

Full reference implementation of the hierarchical columnar network
specified, trained on the CIFAR-10 image dataset from Hugging-Face.

Key features;
- Dense **or** sparse synapse storage (``use_sparse_matrix`` flag)
- Online structural training exactly as requested (random	 selection of
  ``trainN`` neurons - ``trainB`` branches - ``trainS`` segments - ``trainI``
  synapses, skipping already-wired segments).
- Final inference chooses the class/column whose *excitatory* neuron in the
  last layer is most active (ties broken arbitrarily).
- Three NetworkX-based visualisation utilities:
  (a) single-neuron detail; (b) whole-network neuron-level state;
  (c) full synapse-level state (very large - optional).

# Specification (o3 prompt):
please examine this biological neural network learning algorithm hypothesis;
if a neuron (excitatory or inhibitory; E or I) accepts multiple independent sets of inputs (at each dendritic segment) then it must be trained to associate each independent set with the same output (either absolute class target or some intermediary class target).
	how could an intermediary layer neuron class target be selected? the class target could be selected from a previous/different dataset.
		would imply each neuron can only be associated. with a particular dataset and class target.
	how would the class target reward signal be sent to the neuron? possibly via another neuron type eg glial cell
each neuron accepts 1000+ independent sets of 10+ excitatory inputs at each segment (with inhibition mainly occurring at later stage in dendrite ie proximal to soma).
	each neuron may only learn a subset of inputs (requiring many more neurons of the same class target in the layer to learn the rest).
		if not fully connected however then there may be too many combinations of subsets (arbitrary selection of subsets).
		if only 1 neuron per class target then not enough. neurons in layer for next layer to be trained.
	each layer is sparsely activated (a neuron will learn to capture every possible combination of inputs at that layer for the class target in a dataset); imply number of neurons trained per layer >= number class targets.
	perhaps closer dendritic segments learn most common sets of inputs (E or I) for class target and more distal dendrites learn most independent sets of inputs (E or I).
	multiple ANN (artificial) layers could be encoded in same bio neuron.
an inhibitory neuron being trained for a class target could learn different combinations of input that are never active for the class target (ie they are learning a "negative sample" or non-class target sample; see contrastive learning).
	an inhibitory neuron for a given class could simply be trained by connecting all the neurons from previous layer of a different class target.
require parallel processing (hardware accelerated) of layers.
for now assume the lower (conv) layers of a CNN have been trained.
each cortical column could be learning a different dataset (set of classes).
---
This is my draft design plan for training:
assign a set of neurons per layer for each class target being trained.
	but how to connect their inputs? connect only same class active excitatory neurons. during inference however the network will select some out of class neurons, and some in class neurons.
	[NO: in a particular context a neuron signals a class, and in another context it signals a different class; only dendritic segments (approx 20 inputs) pertain to the same class.]
during train select some (or one) neurons to train and some (or one) segments to train (of same class).
	but connect all active neurons in previous layer (of same class)? or only a subset of active neurons?
if during inference more neurons of an incorrect class are active in first hidden layer than for the correct class, it relies on the second hidden layer to detect only common active combinations of first hidden layer (to discriminate correct class).
first hidden layer is sensitive to all input layer neurons during train (not just of same class).
cortical (mini) columns are used to train separate classes.
	can overload columns for training of different classes in different datasets however.
inputs to excitatory neurons occur within class column (detect and connect all active neurons within class column to excitatory neuron segment), but inputs to inhibitory neurons occur between columns (detect and connect all inactive neurons to inhibitory neuron segment).
---
I would like a hardware accelerated pytorch implementation of my draft design plan for training without your modifications (eg "1 Clarify the objective for layer L");
- every layer except for the input layer is split into a set of columns (for each target class in the dataset), each column is split into a set of excitatory (E) or inhibitory (I) neurons, each neuron is split into a set of "branches" (from close to soma to far from soma), each branch is split into a set of segments, and each segment is split into a set of synaptic inputs. The pytorch connection tensor (connectionMatrix) thus has the following dimensions: dim0: layer, dim1: column, dim2:  type, dim3: branch, dim4: segment, dim5: synapticInput.
- Please create two implementations of connectionMatrix (depending on the value of bool useSparseMatrix);
1. useSparseMatrix=False: all synaptic connections will use binary weights (connected/not connected to previous layer neuron), and are fully connected to all neurons on the previous layer.
2. useSparseMatrix=True: the synaptic connections store a set of indices referring to their previous layer neurons.
- The activation of dendritic branches can be added together (AND), where as the activation from dendritic segments are independently selected (OR). Each segment's synaptic inputs must reach a minimum activation threshold to send a dendritic NMDA spike to the soma (eg x out of 20 inputs must be activated).
- Do not hardcode any variables (ie no magic numbers)
---
The pytorch implementation must provide both training and inference code. Use my specification for training the excitatory and inhibitory neurons. 
-  For each training iteration (dataset sample), select trainN excitatory and inhibitory neurons to train per layer, trainB branches to train per neuron, trainS segments per branch, and trainI inputs per segment. Initialise trainN=1, trainB=1, trainS=1, trainI=numberInputsPerSegment. Assume equal numbers of excitatory and inhibitory neurons (and equal numbers being trained).
- Always randomly select neurons (from each neuron type), branches, segments, and synaptic inputs to train.  For now (to simplify the implementation); if a segment is already trained, ignore this and select the next untrained segment (maintain a matrix indicating which segments are available to train).
---
- Please use a real dataset from huggingface to train the network (for now select an image dataset with a large number of inputs; CIFAR-10). 
- The final classification during inference occurs by selecting the final layer column with the most number of activated excitatory neurons.
- provide a complete visualisation of the network using the networkx library to assist manual train/inference debugging, with;
a) a bool option to visualise the excitation state (on/off) of every synaptic input, segment, branch in a select neuron in the network after a sample has been propagated during train/inference.
b) a bool option to visualise the excitation state (on/off) of every neuron in the network after a sample has been propagated during train/inference. Network is drawn with segregated layers and segregated neuron types. The connectivity of every neuron to their previous layer neurons should be drawn (ie draw a line between them).
c) a bool option to visualise the excitation state (on/off) of every synaptic input, segment, branch, and neuron in the network after a sample has been propagated during train/inference. Network is drawn with segregated layers and segregated neuron types. The connectivity of every synaptic input to their previous layer neurons should be drawn (ie draw a line between them).  This bool option (c) will produce a very large graph, and may be too large to display.

"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset

# ----------------------------------------------------------------------
# 1 - Configuration dataclass (all user-settable, no magic numbers)
# ----------------------------------------------------------------------
@dataclass
class ColumnarNetworkConfig:
	# structural knobs
	num_layers: int					   # incl. input layer
	num_columns: int
	num_types: int						# 0 = E, 1 = I
	num_branches: int
	num_segments: int
	num_synapses: int					 # per segment
	input_size: int					   # # input neurons (flattened image)

	# thresholds
	seg_threshold: int					# OR threshold within a segment
	branch_threshold: int				 # AND threshold across branches

	# training hyper-params
	trainN: int = 1					   # neurons/type/layer/sample
	trainB: int = 1
	trainS: int = 1
	trainI: Optional[int] = None		  # defaults to num_synapses

	# misc
	use_sparse_matrix: bool = True		# True - sparse indices
	device: torch.device = torch.device(
		"cuda" if torch.cuda.is_available() else "cpu")

	# reproducibility
	seed: int = 0

	def __post_init__(self):
		torch.manual_seed(self.seed)
		random.seed(self.seed)
		if self.trainI is None:
			self.trainI = self.num_synapses
		assert self.trainI <= self.num_synapses, "trainI > num_synapses"


# ----------------------------------------------------------------------
# 2 - Columnar Network   (inference only)
# ----------------------------------------------------------------------
class ColumnarNetwork(nn.Module):
	"""Layer - Column - Type - Branch - Segment - Synapse hierarchy."""

	def __init__(self, cfg: ColumnarNetworkConfig):
		super().__init__()
		self.cfg = cfg

		self.connection_matrices = nn.ParameterList()
		self.segment_trained = nn.ParameterList()

		for l in range(1, cfg.num_layers):
			prev_size = cfg.input_size if l == 1 else (cfg.num_columns * cfg.num_types)

			if cfg.use_sparse_matrix:
				shape = (cfg.num_columns, cfg.num_types, cfg.num_branches, cfg.num_segments, cfg.num_synapses)
				idx = torch.full(shape, -1, dtype=torch.long, device=cfg.device)
				self.connection_matrices.append(nn.Parameter(idx, requires_grad=False))
			else:
				shape = (cfg.num_columns, cfg.num_types, cfg.num_branches, cfg.num_segments, prev_size)
				dense = torch.zeros(shape, dtype=torch.bool, device=cfg.device)
				self.connection_matrices.append(nn.Parameter(dense, requires_grad=False))

			flag_shape = (cfg.num_columns, cfg.num_types, cfg.num_branches, cfg.num_segments)
			trained = torch.zeros(flag_shape, dtype=torch.bool, device=cfg.device)
			self.segment_trained.append(nn.Parameter(trained, requires_grad=False))

	# --------------------------------------------------------------
	def forward(self, x: torch.Tensor):
		"""x : (batch, input_size)�binary {0,1}"""
		act = x.bool()
		trace: List[torch.Tensor] = [act]

		for l in range(1, self.cfg.num_layers):
			conn = self.connection_matrices[l - 1]
			if self.cfg.use_sparse_matrix:
				act = self._forward_sparse(act, conn)
			else:
				act = self._forward_dense(act, conn)
			trace.append(act)
		return trace[-1][:, :, 0].int(), trace  # excitatory logits, full trace

	# --------------------------------------------------------------
	def _forward_dense(self, prev: torch.Tensor, conn: torch.Tensor):
		# prev : (B, N_prev)
		prev_exp = prev.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
		matched = conn & prev_exp  # broadcast bool AND
		seg_sum = matched.sum(-1)
		seg_on = seg_sum >= self.cfg.seg_threshold
		branch_sum = seg_on.sum(-1)
		branch_on = branch_sum >= self.cfg.branch_threshold
		return branch_on  # (B, C, T)

	# --------------------------------------------------------------
	def _forward_sparse(self, prev: torch.Tensor, idx: torch.Tensor):
		# Ensure prev is 2D (B, N_prev)
		if prev.dim() == 1:
			prev = prev.unsqueeze(0)
		elif prev.dim() > 2:
			prev = prev.view(prev.size(0), -1)
		B, N_prev = prev.shape
		C, T, Br, S, _ = idx.shape
		idx_flat = idx.view(-1, self.cfg.num_synapses)  # (M,K)
		gathered = torch.where(idx_flat >= 0,
							   prev[:, idx_flat.clamp(min=0)],
							   torch.zeros_like(prev[:, :1]))
		gathered = gathered.view(B, C, T, Br, S, self.cfg.num_synapses)
		seg_sum = gathered.sum(-1)
		seg_on = seg_sum >= self.cfg.seg_threshold
		branch_sum = seg_on.sum(-1)
		branch_on = branch_sum >= self.cfg.branch_threshold
		return branch_on


# ----------------------------------------------------------------------
# 3 - Trainer - online structural learning
# ----------------------------------------------------------------------
class ColumnarTrainer:
	def __init__(self, net: ColumnarNetwork, cfg: ColumnarNetworkConfig):
		self.net = net
		self.cfg = cfg
		self.rng = random.Random(cfg.seed)

	# helpers ------------------------------------------------------
	def _untrained_segment(self, layer: int, col: int, typ: int):
		trained_flag = self.net.segment_trained[layer - 1]
		attempts = 0
		while attempts < 1000:
			b = self.rng.randrange(self.cfg.num_branches)
			s = self.rng.randrange(self.cfg.num_segments)
			if not trained_flag[col, typ, b, s]:
				trained_flag[col, typ, b, s] = True
				return b, s
			attempts += 1
		return None  # No untrained segments left

	# --------------------------------------------------------------
	@torch.no_grad()
	def train_sample(self, x: torch.Tensor, class_id: int):
		logits, layers = self.net(x)
		cfg = self.cfg

		# loop over hidden+output layers							
		for l in range(1, cfg.num_layers):
			prev_act = layers[l - 1][0]  # (N_prev) or (C_prev,T_prev)

			# build active / inactive pools ------------------------
			if prev_act.dim() == 1:
				idx_active = prev_act.nonzero(as_tuple=False).flatten().tolist()
				idx_inactive = (prev_act == 0).nonzero(as_tuple=False).flatten().tolist()
				total_prev = prev_act.numel()
			else:
				shape = prev_act.shape
				if len(shape) == 2:
					C_prev, T_prev = shape
				else:
					C_prev, T_prev = shape[0], int(torch.prod(torch.tensor(shape[1:])))
					prev_act = prev_act.view(C_prev, T_prev)
				idx_active, idx_inactive = [], []
				for c in range(C_prev):
					for t in range(T_prev):
						flat = c * T_prev + t
						(idx_active if prev_act[c, t] else idx_inactive).append(flat)
				total_prev = C_prev * T_prev

			# ------------------------------------------------------
			for typ in range(cfg.num_types):		  # 0=E,1=I
				for _ in range(cfg.trainN):
					for _ in range(cfg.trainB):
						for _ in range(cfg.trainS):
							seg_result = self._untrained_segment(l, class_id, typ)
							if seg_result is None:
								continue  # No untrained segments left, skip
							b, s = seg_result
							if typ == 0:
								pool = idx_active
							else:
								pool = idx_inactive or [i for i in range(total_prev) if i not in idx_active]
							if not pool:
								continue  # nothing to wire
							chosen = self.rng.sample(pool * ((cfg.trainI // len(pool)) + 1), cfg.trainI)

							conn = self.net.connection_matrices[l - 1]
							if cfg.use_sparse_matrix:
								seg = conn[class_id, typ, b, s]
								seg[:] = torch.tensor(chosen, dtype=torch.long, device=cfg.device)
							else:
								seg = conn[class_id, typ, b, s]
								seg.zero_()
								seg[chosen] = True


# ----------------------------------------------------------------------
# 4 - Dataset utilities  (load CIFAR-10 + binarise)
# ----------------------------------------------------------------------

def load_cifar10(flatten: bool = True):
	ds = load_dataset("cifar10", split="train")
	transf = transforms.Compose([
		transforms.ToTensor(),		  # 0-1 float, C�32�32
		transforms.Grayscale(),		 # 1�32�32
		transforms.Lambda(lambda x: (x > 0.5).bool()),
	])
	images, labels = [], []
	for sample in ds:
		img = transf(sample["img"])	 # 1�32�32 bool
		if flatten:
			img = img.view(-1)			# 1024 bool
		images.append(img)
		labels.append(int(sample["label"]))
	images = torch.stack(images)		  # (N, 1024)
	labels = torch.tensor(labels)
	return images, labels


# ----------------------------------------------------------------------
# 5 - NetworkX visualisation helpers
# ----------------------------------------------------------------------

def _color(active: bool):
	return "red" if active else "lightgray"


def visualise_neuron(net: ColumnarNetwork, trace: List[torch.Tensor],
					 layer: int, column: int, typ: int, viz_inputs=True):
	"""Option (a) : show synapses, segments, branches for ONE neuron."""
	cfg = net.cfg
	G = nx.DiGraph()
	pos: dict[str, tuple[float, float]] = {}

	# nodes --------------------------------------------------------
	neuron_node = ("L{}C{}T{}".format(layer, column, typ))
	G.add_node(neuron_node, color="black")
	pos[neuron_node] = (0.0, 0.0)

	for b in range(cfg.num_branches):
		# compute segment activations for this branch
		seg_actives = []
		for s in range(cfg.num_segments):
			# compute segment activation
			seg_active = False
			if cfg.use_sparse_matrix:
				idx = net.connection_matrices[layer - 1][column, typ, b, s]
				valid = idx[idx >= 0]
				if valid.numel():
					prev_flat = trace[layer - 1][0].view(-1)
					hits = prev_flat[valid]
					seg_active = hits.sum().item() >= cfg.seg_threshold
			else:
				dense = net.connection_matrices[layer - 1][column, typ, b, s]
				hits = dense & trace[layer - 1][0].view(-1)
				seg_active = hits.sum().item() >= cfg.seg_threshold
			seg_actives.append(seg_active)
		# determine branch activation from segments
		branch_active = sum(seg_actives) >= cfg.branch_threshold
		branch_node = f"B{b}"
		G.add_node(branch_node, color=_color(bool(branch_active)))
		G.add_edge(branch_node, neuron_node)
		pos[branch_node] = (1.0, (b + 1) / (cfg.num_branches + 1) - 0.5)

		for s, seg_active in enumerate(seg_actives):
			seg_node = f"B{b}S{s}"
			G.add_node(seg_node, color=_color(seg_active))
			G.add_edge(seg_node, branch_node)
			branch_y = pos[branch_node][1]
			pos[seg_node] = (2.0, branch_y + (s + 1) / (cfg.num_segments + 1) * 0.5)

			if viz_inputs:
				# draw synaptic inputs
				if cfg.use_sparse_matrix:
					valid = idx[idx >= 0]
					for syn_i, prev_idx in enumerate(valid.tolist()):
						syn_node = f"inp{prev_idx}"
						prev_active = trace[layer - 1][0].view(-1)[prev_idx].item()
						G.add_node(syn_node, color=_color(prev_active))
						G.add_edge(syn_node, seg_node)
						pos[syn_node] = (3.0, float(prev_idx))
				else:
					dense = net.connection_matrices[layer - 1][column, typ, b, s]
					act_prev = trace[layer - 1][0].view(-1)
					conn_idx = dense.nonzero(as_tuple=False).flatten().tolist()
					for prev_idx in conn_idx:
						syn_node = f"inp{prev_idx}"
						prev_active = act_prev[prev_idx].item()
						G.add_node(syn_node, color=_color(prev_active))
						G.add_edge(syn_node, seg_node)
						pos[syn_node] = (3.0, float(prev_idx))

	# draw ---------------------------------------------------------
	colors = [G.nodes[n]["color"] for n in G.nodes]
	nx.draw(G, pos, node_color=colors, with_labels=False, node_size=200)
	plt.title("Neuron detail L{} C{} T{}".format(layer, column, typ))
	plt.show()



def visualise_network(net: ColumnarNetwork, trace: List[torch.Tensor],
					  synapse_level: bool = False):
	"""Option (b)/(c) : show all neurons; optionally down to synapses."""
	cfg = net.cfg
	G = nx.DiGraph()
	pos = {}

	# build graph layer-by-layer ----------------------------------
	for l in range(1, cfg.num_layers):
		prev_flat = trace[l - 1][0].view(-1)
		for c in range(cfg.num_columns):
			for t in range(cfg.num_types):
				neuron_node = f"L{l}C{c}T{t}"
				# neuron active if any branch is active
				active = trace[l][0, c, t].any().item()
				G.add_node(neuron_node, color=_color(active))
				# position neuron on grid
				pos[neuron_node] = (l * 3, c * cfg.num_types + t)
				# connect from branch container (for visual separation)
				for b in range(cfg.num_branches):
					branch_node = f"L{l}C{c}T{t}B{b}"
						# position branch on grid
					pos[branch_node] = (l * 3 + 1, c * cfg.num_types + t + (b + 1) / (cfg.num_branches + 1) - 0.5)
					# compute segment activations for this branch
					seg_actives = []
					for s in range(cfg.num_segments):
						# compute segment activation
						seg_active = False
						if cfg.use_sparse_matrix:
							idx = net.connection_matrices[l - 1][c, t, b, s]
							valid = idx[idx >= 0]
							if valid.numel():
								prev_flat = trace[l - 1][0].view(-1)
								hits = prev_flat[valid]
								seg_active = hits.sum().item() >= cfg.seg_threshold
						else:
							dense = net.connection_matrices[l - 1][c, t, b, s]
							hits = dense & trace[l - 1][0].view(-1)
							seg_active = hits.sum().item() >= cfg.seg_threshold
						seg_actives.append(seg_active)
					# determine branch activation from segments
					branch_active = sum(seg_actives) >= cfg.branch_threshold
					G.add_node(branch_node, color=_color(bool(branch_active)))
					G.add_edge(branch_node, neuron_node)
					if not synapse_level:
						continue
					for s, seg_active in enumerate(seg_actives):
						seg_node = f"L{l}C{c}T{t}B{b}S{s}"
						G.add_node(seg_node, color=_color(seg_active))
						# position segment on grid
						pos[seg_node] = (l*3+2, c*cfg.num_types + t + (b + (s+1)/(cfg.num_segments+1)) - 0.5)
						G.add_edge(seg_node, branch_node)
						# synapse nodes on the right
						prev_flat = trace[l-1][0].view(-1)
						# gather connection indices
						if cfg.use_sparse_matrix:
							_conn = net.connection_matrices[l-1][c, t, b, s]
							conn_idx = _conn[_conn >= 0].tolist()
						else:
							conn_idx = net.connection_matrices[l-1][c, t, b, s].nonzero(as_tuple=False).flatten().tolist()
						for prev_idx in conn_idx:
							syn_node = f"in{prev_idx}"
							if not G.has_node(syn_node):
								prev_active = prev_flat[prev_idx].item()
								G.add_node(syn_node, color=_color(bool(prev_active)))
								# position synapse on grid
								pos[syn_node] = (l*3+3, prev_idx)
							G.add_edge(syn_node, seg_node)
	# draw ---------------------------------------------------------
	# draw with explicit positions
	colors = [G.nodes[n]["color"] for n in G.nodes]
	nx.draw(G, pos, node_color=colors, with_labels=False, node_size=50, edge_color="gray")
	plt.title("Full network ({} level)".format("synapse" if synapse_level else "neuron"))
	plt.show()


# ----------------------------------------------------------------------
# 6 - Main - CLI wrapper
# ----------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(description="Columnar excitatory/inhibitory (EI) neuron network trainer (CIFAR-10)")
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--train-samples", type=int, default=1000)
	parser.add_argument("--use-sparse", action="store_true")
	parser.add_argument("--viz-neuron", action="store_true")
	parser.add_argument("--viz-network", action="store_true")
	parser.add_argument("--viz-full", action="store_true")
	args = parser.parse_args()

	# load data ----------------------------------------------------
	images, labels = load_cifar10()

	cfg = ColumnarNetworkConfig(
		num_layers=3,
		num_columns=10,
		num_types=2,
		num_branches=4,
		num_segments=4,
		num_synapses=20,
		input_size=1024,
		seg_threshold=10,
		branch_threshold=2,
		trainN=1,
		trainB=1,
		trainS=1,
		trainI=None,
		use_sparse_matrix=args.use_sparse,
		seed=0,
	)

	net = ColumnarNetwork(cfg).to(cfg.device)
	trainer = ColumnarTrainer(net, cfg)

	# simple online training --------------------------------------
	for i in range(min(args.train_samples, images.size(0))):
		x = images[i : i + 1].to(cfg.device)
		class_id = labels[i].item()
		trainer.train_sample(x, class_id)

	# inference demo ----------------------------------------------
	net.eval()
	test_x = images[-1:].to(cfg.device)
	test_label = labels[-1].item()
	excitatory_logits, trace = net(test_x)
	pred_class = excitatory_logits.sum(-1).argmax().item()
	print(f"Ground truth : {test_label} | Predicted: {pred_class}")

	# visualisation options ---------------------------------------
	if args.viz_neuron:
		visualise_neuron(net, trace, layer=1, column=pred_class, typ=0)
	if args.viz_network:
		visualise_network(net, trace, synapse_level=False)
	if args.viz_full:
		visualise_network(net, trace, synapse_level=True)


if __name__ == "__main__":
	main()

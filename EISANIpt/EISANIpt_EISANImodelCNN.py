"""EISANIpt_EISANImodelCNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model CNN

"""

import os
import sys
import importlib.util
import torch
from torch import nn
from ANNpt_globalDefs import *
import itertools
import torch.nn.functional as F
import math 


if(EISANICNNarchitectureSparseRandom):
	import EISANIpt_EISANImodelSummation
	
	class EISANICNNconfig():
		def __init__(self, hiddenLayerSize, numberOfConvlayers, numberOfSynapsesPerSegment, layer_input_sizes, sani_per_kernel, kernels_per_layer):
			print("layer_input_sizes = ", layer_input_sizes)
			if layer_input_sizes is None:
				raise ValueError("layer_input_sizes must be provided for sparse CNN initialisation")
			if len(layer_input_sizes) != numberOfConvlayers:
				raise ValueError("layer_input_sizes length must match numberOfConvlayers")
			self.numberOfConvlayers = numberOfConvlayers
			self.hiddenLayerSize = hiddenLayerSize
			self.numberOfSynapsesPerSegment = numberOfSynapsesPerSegment
			self.layer_input_sizes = layer_input_sizes
			self.sani_per_kernel = sani_per_kernel
			self.kernels_per_layer = kernels_per_layer
		
	class EISANICNNmodel(nn.Module):
		"""Custom CNN binary neural network implementing the EISANI specification."""

		def __init__(self, config: EISANICNNconfig) -> None:
			super().__init__()

			self.config = config
			self.numberUniqueHiddenLayers = config.numberOfConvlayers
			self.layer_input_sizes = config.layer_input_sizes
			self.sani_per_kernel = config.sani_per_kernel
			self.kernels_per_layer = config.kernels_per_layer

			self.hiddenConnectionMatrix = [[] for _ in range(self.numberUniqueHiddenLayers)]
			if useEIneurons:
				self.hiddenConnectionMatrixExcitatory = [[] for _ in range(self.numberUniqueHiddenLayers)]
				self.hiddenConnectionMatrixInhibitory = [[] for _ in range(self.numberUniqueHiddenLayers)]
			else:
				self.hiddenConnectionMatrixExcitatory = []
				self.hiddenConnectionMatrixInhibitory = []

			for hiddenLayerIdx in range(self.numberUniqueHiddenLayers):
				prevSize = self.layer_input_sizes[hiddenLayerIdx]
				for segmentIdx in range(numberOfSegmentsPerNeuron):
					if useEIneurons:
						if useInhibition:
							excitSize = config.hiddenLayerSize // 2
							inhibSize = config.hiddenLayerSize - excitSize
						else:
							excitSize = config.hiddenLayerSize
							inhibSize = 0
						excMat = EISANIpt_EISANImodelSummation.initialise_layer_weights(self, excitSize, prevSize, hiddenLayerIdx)
						self.hiddenConnectionMatrixExcitatory[hiddenLayerIdx].append(excMat)
						if inhibSize > 0:
							inhMat = EISANIpt_EISANImodelSummation.initialise_layer_weights(self, inhibSize, prevSize, hiddenLayerIdx)
							self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(inhMat)
						else:
							placeholder_dtype = torch.bool if useSparseHiddenMatrix else torch.int8
							placeholder = torch.empty((0, prevSize), dtype=placeholder_dtype, device=device)
							self.hiddenConnectionMatrixInhibitory[hiddenLayerIdx].append(placeholder)
					else:
						mat = EISANIpt_EISANImodelSummation.initialise_layer_weights(self, config.hiddenLayerSize, prevSize, hiddenLayerIdx)
						self.hiddenConnectionMatrix[hiddenLayerIdx].append(mat)

		def summationSANIpassCNNlayer(self, prevActivation, layerIdCNN):
			uniqueLayerIndex = layerIdCNN
			if useEIneurons:
				aExc, aInh = EISANIpt_EISANImodelSummation.compute_layer_EI(self, uniqueLayerIndex, prevActivation, device)
				currentActivation = torch.cat([aExc, aInh], dim=1)
			else:
				currentActivation = EISANIpt_EISANImodelSummation.compute_layer_standard(self, uniqueLayerIndex, prevActivation, device)
			return currentActivation	

# ---------------------------------------------------------
# IMAGE helpers (init & propagation)
# ---------------------------------------------------------

def init_conv_layers(self) -> None:
	assert CNNstride == 1, "Only stride-1 kernels supported in this implementation"

	if(EISANICNNarchitectureSparseRandom):
		assert EISANICNNpaddingPolicy == 'same', "Sparse random CNN currently supports 'same' padding only"
		self._EISANICNN_layer_in_channels = []
		self._EISANICNN_layer_input_sizes = []
		self._EISANICNN_out_channels = []
		self._EISANICNN_sani_per_kernel = EISANICNNkernelSizeSANI
		self._EISANICNN_total_neurons_per_layer = EISANICNNnumberKernels * EISANICNNkernelSizeSANI
		in_channels = numberInputImageChannels * EISANICNNcontinuousVarEncodingNumBits
		for _ in range(self.config.numberOfConvlayers):
			self._EISANICNN_layer_in_channels.append(in_channels)
			self._EISANICNN_out_channels.append(EISANICNNnumberKernels)
			patch_size = in_channels * CNNkernelSize * CNNkernelSize
			self._EISANICNN_layer_input_sizes.append(patch_size)
			# Preserve the full SANI neuron count as the channel width for the next layer.
			in_channels = self._EISANICNN_total_neurons_per_layer
	elif(EISANICNNarchitectureDenseRandom):
		self.cnn_dense_layers = nn.ModuleList()
		self._dense_random_out_channels = []
		in_channels = numberInputImageChannels*EISANICNNcontinuousVarEncodingNumBits
		base_channels = 64
		max_channels = 512
		for layerIdx in range(self.config.numberOfConvlayers):
			scale = layerIdx // max(1, EISANICNNmaxPoolEveryQLayers)
			out_channels = min(base_channels * (2 ** scale), max_channels)
			padding = CNNkernelSize // 2 if EISANICNNpaddingPolicy == 'same' else 0
			conv = nn.Conv2d(in_channels, out_channels, kernel_size=CNNkernelSize, stride=1, padding=padding, bias=False)
			nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
			self.cnn_dense_layers.append(conv)
			self._dense_random_out_channels.append(out_channels)
			in_channels = out_channels
	elif(EISANICNNarchitectureDensePretrained):
		module_path = os.path.join(os.path.dirname(__file__), 'Resnet18-breakaway', 'resnet18_breakaway.py')
		spec = importlib.util.spec_from_file_location('resnet18_breakaway', module_path)
		if spec is None or spec.loader is None:
			raise RuntimeError('Unable to load ResNet18 backbone description')
		resnet_module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = resnet_module
		spec.loader.exec_module(resnet_module)
		backbone = resnet_module.ResNet18Breakaway(num_classes=numberOfClasses)
		weights_path = os.path.join(os.path.dirname(__file__), 'Resnet18-breakaway', 'resnet18_full.pth')
		if not os.path.exists(weights_path):
			raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
		state = torch.load(weights_path, map_location=device)
		if isinstance(state, dict) and 'state_dict' in state:
			state = state['state_dict']
		if all(k.startswith('module.') for k in state.keys()):
			state = {k[len('module.'):]: v for k, v in state.items()}
		filtered_state = {k: v for k, v in state.items() if not k.startswith('bk_heads.')}
		excluded = set(state.keys()) - set(filtered_state.keys())
		if excluded:
			print(f"[EISANICNN] Skipping breakaway head params: {sorted(excluded)}")
		missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)
		allowed_missing_prefixes = ('bk_heads.',)
		missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
		if unexpected:
			print(f"[EISANICNN] Unexpected keys after load_state_dict: {unexpected}")
		if missing:
			print(f"[EISANICNN] Missing keys after load_state_dict: {missing}")
		backbone.to(device)
		backbone.eval()
		for param in backbone.parameters():
			param.requires_grad_(False)
		self.cnn_pretrained = backbone
		dummy = torch.zeros(1, numberInputImageChannels*EISANICNNcontinuousVarEncodingNumBits, inputImageHeight, inputImageWidth, device=device)
		with torch.no_grad():
			features = backbone._forward_body(dummy)
			pooled = backbone.gap(features)
		self._pretrained_feature_shape = pooled.shape[1:]
		self.encodedFeatureSize = pooled.view(1, -1).shape[1] * EISANITABcontinuousVarEncodingNumBitsAfterCNN
		return
	elif(EISANICNNarchitectureDivergeAllKernelPermutations or EISANICNNarchitectureDivergeLimitedKernelPermutations):
		if(EISANICNNarchitectureDivergeAllKernelPermutations):
			"""
			Initialise binary kernel bank (3  3, 1) and compute flattened
			output size after `numberOfConvlayers` (+ optional max-pooling).
			Sets:
			 self.convKernels	 (511,1,3,3) binary +1/0 (stored as float for conv2d)
			 self.encodedFeatureSize
			"""
			assert CNNkernelSize == 3
			# --- generate every possible 3 x 3 BINARY mask (+1 / -1)
			all_patterns = torch.tensor(list(itertools.product([-1, 1], repeat=CNNkernelSize * CNNkernelSize)), dtype=torch.int8)
			all_patterns = all_patterns[(all_patterns.sum(dim=1) > 0)]	# drop the all-zero mask (would match everywhere)
		elif(EISANICNNarchitectureDivergeLimitedKernelPermutations):		
			ksz = int(CNNkernelSize)
			# --- Arbitrary-o oriented ksz�ksz *float* kernels on the given device.
			# Grid: center at (0,0), sample at X,Y E {-1,0,1}.  0_k = 2pik/o.
			# Define u (axis-aligned) and v (perpendicular) via rotation:
			#   u =  X*cos0 + Y*sin0
			#   v = -X*sin0 + Y*cos0
			# We shape odd-symmetric filters with tanh(v/\u03c4), localized by a Gaussian envelope.
			# Ternary mode additionally gates with a soft zero-band around v\u22480.
			o = EISANICNNnumberKernelOrientations
			assert o >= 1, "EISANICNN: orientations (o) must be >= 1"
			sigma = EISANICNNsigma
			tau = EISANICNNtau
			band = EISANICNNkernelZeroBandHalfWidth

			half = (ksz - 1) / 2.0
			coords = torch.arange(ksz, device=device, dtype=torch.float32) - half
			Y, X = torch.meshgrid(coords, coords, indexing='ij')  # (ksz,ksz)

			angles = torch.arange(o, device=device, dtype=torch.float32) * (2*torch.pi / o)  # [0,2pi)
			s = angles.sin().view(o, 1, 1)  # (o,1,1)
			c = angles.cos().view(o, 1, 1)  # (o,1,1)
			u = (X*c + Y*s)                 # (o,3,3) after broadcasting
			v = (-X*s + Y*c)                # perpendicular distance (o,ksz,ksz)
			env = torch.exp(-(X**2 + Y**2) / (2.0 * (sigma**2)))  # (ksz,ksz)
			env = env.view(1, ksz, ksz).expand(o, -1, -1)         # (o,ksz,ksz)

			K_total = 0
			core_edge = -torch.tanh(v / tau)	# (kept for possible future use)
			banks = []
			# hard threshold (in grid units) so at theta=0 and ksz=3 we get:
			# Edge:   [+1 +1 +1; 0 0 0; -1 -1 -1]
			# Corner: [+1 +1 +1; -1 -1 +1; -1 -1 +1]
			t = float(globals().get('EISANICNNbandThreshold', 0.5))
			if EISANICNNkernelEdges:
				if(EISANICNNkernelEdgesSharp):
					# Oriented EDGES (hard-banded): +1 / 0 / -1 by perpendicular distance v
					if EISANICNNkernelEdgesTernary:
						pos = (v >=  t).to(v.dtype)
						zer = (v.abs() < t).to(v.dtype)
						neg = (v <= -t).to(v.dtype)
						k_edges = (+1.0 * pos) + (0.0 * zer) + (-1.0 * neg)
					else:
						# binary edge (no zero band) \u2014 include the tie (v\u22480) on the positive side
						eps = EISANICNNedgeTieEps = 1e-6
						k_edges = torch.where(v >= -eps, torch.tensor(1.0, device=v.device), torch.tensor(-1.0, device=v.device))
					# no envelope \u21d2 rows/columns stay uniform within a band
					k_edges = k_edges - k_edges.mean(dim=(1,2), keepdim=True)
					k_edges = k_edges / (k_edges.norm(dim=(1,2), keepdim=True) + 1e-8)
				else:
					# Oriented EDGES (existing)
					if EISANICNNkernelEdgesTernary:
						k_edges = env * core_edge
					else:
						printe("!EISANICNNkernelEdgesTernary not currently supported")
					k_edges = k_edges - k_edges.mean(dim=(1,2), keepdim=True)
					k_edges = k_edges / (k_edges.norm(dim=(1,2), keepdim=True) + 1e-8)
				banks.append(k_edges)						# (o,ksz,ksz)
				K_total = K_total+o
			if EISANICNNkernelCorners:
				# Oriented CORNERS (L-junction): +1 if (u >= t) OR (v >= t), else -1
				pos = ((u >= t) | (v >= t)).to(v.dtype)
				k_corners = torch.where(pos > 0, torch.tensor(1.0, device=v.device), torch.tensor(-1.0, device=v.device))
				k_corners = k_corners - k_corners.mean(dim=(1,2), keepdim=True)
				k_corners = k_corners / (k_corners.norm(dim=(1,2), keepdim=True) + 1e-8)
				banks.append(k_corners)
				K_total = K_total+o
			if EISANICNNkernelCentroids:
				# CENTROIDS (isotropic center-surround, not oriented) \u2014 1 kernel
				sigma_c = EISANICNNcentroidSigma
				gamma = EISANICNNcentroidScaleRatio
				sigma_s = sigma_c * gamma
				Gc = torch.exp(-(X**2 + Y**2) / (2.0 * (sigma_c**2)))	# (ksz,ksz)
				Gs = torch.exp(-(X**2 + Y**2) / (2.0 * (sigma_s**2)))	# (ksz,ksz)
				lam = (Gc.sum() / (Gs.sum() + 1e-12))
				k_cent = (Gc - lam * Gs).view(1, ksz, ksz)			# (1,ksz,ksz)
				k_cent = k_cent - k_cent.mean(dim=(1,2), keepdim=True)
				k_cent = k_cent / (k_cent.norm(dim=(1,2), keepdim=True) + 1e-8)
				banks.append(k_cent)
				K_total = K_total+1

			assert len(banks) > 0, "EISANICNN: enable at least one kernel family (Edges/Corners/Hooks/Centroids)"
			kbank = torch.cat(banks, dim=0)					# (K_total,ksz,ksz)
			all_patterns = kbank.view(-1, ksz * ksz)
		
		# --- pretty-print kernels for manual review (prints all by default) ---
		if debugEISANICNNprintKernels:
			ksz = int(CNNkernelSize)
			kernels = all_patterns.view(-1, ksz, ksz).detach().float().cpu()
			K = kernels.shape[0]
			print(f"[EISANICNN] Generated {K} kernels ({ksz}x{ksz}):")
			for idx in range(K):
				print(f"kernel {idx+1}/{K}:")
				for r in range(ksz):
					row = " ".join(f"{v:+.3f}" for v in kernels[idx, r].tolist())
					print("\t" + row)
				print()
		self.convKernels = all_patterns.view(-1, 1, ksz, ksz).float().to(device)	# (outCh,inCh,H,W))		#EISANICNNarchitectureDivergeAllKernelPermutations: (2**9)-1=511 out-channel
		self.convKernels = self.convKernels.float()	#torch.conv2d currently requires float
		print("self.convKernels.shape = ", self.convKernels.shape)
		
	
	# --- compute flattened size after N conv (+max-pool) layers
	H = inputImageHeight
	W = inputImageWidth
	ch = numberInputImageChannels*EISANICNNcontinuousVarEncodingNumBits
	print(f"conv0: channels={ch}, H={H}, W={W}")
	for layerIdx in range(self.config.numberOfConvlayers):
		if EISANICNNpaddingPolicy=='same':
			# 3x3 stride-1, same padding -> H,W unchanged
			pass
		else:
			# 'valid' conv (no padding)	# stride 1 conv
			H = H - CNNkernelSize + 1
			W = W - CNNkernelSize + 1
		if CNNmaxPool and ((layerIdx + 1) % EISANICNNmaxPoolEveryQLayers == 0):
			# use *ceil* division (same as F.max_pool2d(..., ceil_mode=True))
			H = (H + 1) // 2	#orig: H //= 2
			W = (W + 1) // 2	#orig: W //= 2
		if(EISANICNNarchitectureDivergeAllKernelPermutations):
			ch *= (2**9)-1							# each conv multiplies channels by 511
		elif(EISANICNNarchitectureDivergeLimitedKernelPermutations):
			ch *= K_total	#eg each conv multiplies channels by EISANICNNnumberKernelOrientations
		elif(EISANICNNarchitectureSparseRandom):
			ch = self._EISANICNN_total_neurons_per_layer
		elif(EISANICNNarchitectureDenseRandom):
			ch = self._dense_random_out_channels[layerIdx]
		elif(EISANICNNarchitectureDensePretrained):
			pass
		
		print(f"conv{layerIdx+1}: channels={ch}, H={H}, W={W}")
		
	encodedFeatureSizeMax = ch * H * W * EISANITABcontinuousVarEncodingNumBitsAfterCNN
	if(EISANICNNdynamicallyGenerateFFInputFeatures):
		self.encodedFeatureSize = encodedFeatureSizeDefault	#input linear layer encoded features are dynamically generated from historic active neurons in final CNN layer
	else:
		self.encodedFeatureSize = encodedFeatureSizeMax


if(EISANICNNarchitectureSparseRandom):
	def propagate_conv_layers_sparse_random(self, x: torch.Tensor) -> torch.Tensor:
		if not hasattr(self, 'EISANICNNmodel'):
			raise AttributeError('Sparse random CNN model is not initialised. Ensure init_conv_layers and EISANICNNmodel setup completed.')
		if not hasattr(self, '_EISANICNN_layer_input_sizes'):
			raise AttributeError('Sparse random CNN metadata missing. Call init_conv_layers before propagation.')
		z = x.to(torch.float32)
		batch_size = z.size(0)
		height = z.size(2)
		width = z.size(3)
		padding = CNNkernelSize // 2 if EISANICNNpaddingPolicy == 'same' else 0
		for layer_idx in range(self.config.numberOfConvlayers):
			patches = F.unfold(z, kernel_size=CNNkernelSize, padding=padding, stride=1)
			patch_size = self._EISANICNN_layer_input_sizes[layer_idx]
			patches = patches.transpose(1, 2).reshape(-1, patch_size)
			prev_activation = patches.to(torch.int8)
			layer_output = self.EISANICNNmodel.summationSANIpassCNNlayer(prev_activation, layer_idx)
			total_neurons = self.EISANICNNmodel.config.hiddenLayerSize
			layer_output = layer_output.view(batch_size, height * width, total_neurons)
			layer_output = layer_output.view(
				batch_size,
				height,
				width,
				self._EISANICNN_out_channels[layer_idx],
				self._EISANICNN_sani_per_kernel,
			)
			layer_active = layer_output != 0
			prev_dtype = z.dtype
			z = layer_active.reshape(batch_size, height, width, total_neurons)
			z = z.permute(0, 3, 1, 2).to(prev_dtype)
			if EISANICNNactivationFunction:
				z = (z > 0).to(z.dtype)
			if CNNmaxPool and ((layer_idx + 1) % EISANICNNmaxPoolEveryQLayers == 0):
				z = F.max_pool2d(z, kernel_size=2, stride=2, ceil_mode=True)
				z = (z > 0).to(z.dtype)
				height = (height + 1) // 2
				width = (width + 1) // 2
			height, width = z.size(2), z.size(3)
		flat = z.reshape(z.size(0), -1)
		return flat.to(torch.int8)
elif(EISANICNNarchitectureDenseRandom):
	def propagate_conv_layers_dense_random(self, x: torch.Tensor) -> torch.Tensor:
		if not hasattr(self, 'cnn_dense_layers'):
			raise AttributeError('Dense random CNN layers are not initialised. Call init_conv_layers first.')
		z = x.to(torch.float32)
		for layer_idx, conv in enumerate(self.cnn_dense_layers):
			z = conv(z)
			if EISANICNNactivationFunction:
				z = F.relu(z)
			if CNNmaxPool and ((layer_idx + 1) % EISANICNNmaxPoolEveryQLayers == 0):
				z = F.max_pool2d(z, kernel_size=2, stride=2, ceil_mode=True)
		return z.view(z.size(0), -1)
elif(EISANICNNarchitectureDensePretrained):
	def propagate_conv_layers_dense_pretrained(self, x: torch.Tensor) -> torch.Tensor:
		if not hasattr(self, 'cnn_pretrained'):
			raise AttributeError('Pretrained CNN backbone not initialised. Call init_conv_layers first.')
		backbone = self.cnn_pretrained
		device_backbone = next(backbone.parameters()).device
		z = x.to(device_backbone, dtype=torch.float32)
		features = backbone._forward_body(z)
		pooled = backbone.gap(features)
		flat = pooled.view(pooled.size(0), -1)
		if flat.device != x.device:
			flat = flat.to(x.device)
		return flat
elif(EISANICNNarchitectureDivergeAllKernelPermutations):

	def propagate_conv_layers_diverge_all_kernel_permutations(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Full image pipeline:
		 - threshold input channels at `EISANICNNinputChannelThreshold`
		 - repeat {conv -> (optional) max-pool} `numberOfConvlayers` times
		 - flatten to (batch, -1) int8
		"""
		if(EISANICNNoptimisationAssumeInt8):
			z = x.to(torch.int8)
		else:
			z = x.float()
		b_idx, c_idx, B, C = (None, None, None, None)
		for convLayerIndex in range(self.config.numberOfConvlayers):
			if EISANICNNoptimisationSparseConv and convLayerIndex > 0:
				z, b_idx, c_idx, B, C = sparse_conv(self, convLayerIndex, z, b_idx, c_idx, B, C)
			else:
				z, B, C = dense_conv(self, z)
			if CNNmaxPool and ((convLayerIndex + 1) % EISANICNNmaxPoolEveryQLayers == 0):
				if EISANICNNoptimisationSparseConv and convLayerIndex > 0:
					z = sparse_maxpool2d(self, z, kernel_size=2, stride=2)
				else:
					z = F.max_pool2d(z.float(), kernel_size=2, stride=2)
					if(EISANICNNoptimisationAssumeInt8):
						z = z.to(torch.int8)

		if EISANICNNoptimisationSparseConv and self.config.numberOfConvlayers>1:
			if(EISANICNNdynamicallyGenerateFFInputFeatures):
				linearInput = dynamicallyGenerateLinearInputFeaturesVectorised(self, z, b_idx, c_idx, B, C)
			else:
				linearInput = sparse_to_dense(self, z, b_idx, c_idx, B, C)
		else:
			if(EISANICNNdynamicallyGenerateFFInputFeatures):
				printe("EISANICNNdynamicallyGenerateFFInputFeatures requires EISANICNNoptimisationSparseConv and numberOfConvlayers > 1")
			else:
				linearInput = z
		linearInput = linearInput.view(linearInput.size(0), -1)			# (batch, encodedFeatureSize)	#flatten for linear layers
		return linearInput


	def dynamicallyGenerateLinearInputFeaturesVectorised(				# pylint: disable=too-many-arguments
			self,
			CNNoutputLayerFeatures: torch.Tensor,	# (N_active, H, W) sparse activations
			b_idx: torch.Tensor,					# (N_active,) batch index per slice
			c_idx: torch.Tensor,					# (N_active,) channel index per slice
			B: int,								# batch-size
			C: int								# total CNN channels (active+inactive)
	) -> torch.Tensor:
		"""
		Vectorised: converts sparse CNN activations into a binary matrix of shape
		(B, self.encodedFeatureSize) with no Python-level loops.

		Internals
		---------
		• self.cnn_to_linear_map : 1-D LongTensor of length C*H*W
	    	- maps each flat CNN feature index → fixed linear-layer column (or -1)
		• self.nextLinearCol     : scalar Python int = first free column index
		These two tensors are created lazily on the first call.
		"""
		device = CNNoutputLayerFeatures.device
		N_active, H, W = CNNoutputLayerFeatures.shape

		# ------------------------------------------------------------------ #
		# 0. Lazy initialisation of the mapping tensors (vectorised state)    #
		# ------------------------------------------------------------------ #
		num_possible_features = C * H * W
		if not hasattr(self, "cnn_to_linear_map") or self.cnn_to_linear_map.numel() != num_possible_features:
			self.cnn_to_linear_map = torch.full((num_possible_features,), -1, dtype=torch.long, device=device)
			self.nextLinearCol = 0												# first free column

		if(debugEISANICNNdynamicallyGenerateLinearInputFeatures):
			linearInput = sparse_to_dense(self, CNNoutputLayerFeatures, b_idx, c_idx, B, C)
			printf("linearInput.shape = ", linearInput.shape)
			numActive = (linearInput > 0).sum()
			printf("numActive = ", numActive)
			printf("nextLinearCol = ", self.nextLinearCol)

		# ------------------------------------------------------------------ #
		# 1. Find (slice-idx, h, w) of *non-zero* positions in each map      #
		# ------------------------------------------------------------------ #
		coords = (CNNoutputLayerFeatures != 0).nonzero(as_tuple=False)			# (K, 3)
		if coords.numel() == 0:
			return torch.zeros(B, self.encodedFeatureSize, device=device)

		n, h, w = coords[:, 0], coords[:, 1], coords[:, 2]						# (K,)

		# ------------------------------------------------------------------ #
		# 2. Compute flat CNN index  idx = c*H*W + h*W + w                    #
		# ------------------------------------------------------------------ #
		c = c_idx[n]															# channel for coord
		b = b_idx[n]															# batch-sample
		flat_idx = (c * H * W) + (h * W) + w									# (K,)

		# ------------------------------------------------------------------ #
		# 3. Allocate columns for *new* CNN features (pure tensor ops)        #
		# ------------------------------------------------------------------ #
		current_cols = self.cnn_to_linear_map[flat_idx]							# (K,) may be −1
		new_mask = current_cols.eq(-1)											# bool (K,)
		if new_mask.any():
			# Unique flat indices that still need a column
			unique_new = flat_idx[new_mask].unique()							# (M,)
			M = unique_new.numel()
			if self.nextLinearCol + M > self.encodedFeatureSize:
				raise RuntimeError(f"encodedFeatureSize={self.encodedFeatureSize} exhausted: need {M} more columns.")

			new_cols = torch.arange(self.nextLinearCol, self.nextLinearCol + M, device=device, dtype=torch.long,)	# (M,)
			# Vectorised scatter: fill look-up table for all new indices
			self.cnn_to_linear_map[unique_new] = new_cols
			self.nextLinearCol += M											# advance pointer

			# Re-gather to get columns for *all* flat_idx (old + newly added)
			current_cols = self.cnn_to_linear_map[flat_idx]						# (K,)

		# ------------------------------------------------------------------ #
		# 4. Build the output matrix in one scatter call                      #
		# ------------------------------------------------------------------ #
		linearInput = torch.zeros(B, self.encodedFeatureSize, device=device)
		linearInput.index_put_((b, current_cols), torch.ones_like(b, dtype=linearInput.dtype))

		return linearInput



	def sparse_conv(self, convLayerIndex, x, b_idx, c_idx, B, C) -> torch.Tensor:

		'''
		select parts of the orig tensor x based on any result, returning xSubset, xSubsetIndices
		'''

		if(convLayerIndex == 0):
			print("sparse_conv warning: convLayerIndex == 0, function designed for sparse input") 
		if(convLayerIndex <= 1):
			prevConvOut = x
			B, C, *spatial = prevConvOut.shape			# save spatial dims
			prevConvOutFlat = prevConvOut.view(B, C, -1)				# [B, C, W*H]	 #flatten width/height dim
			prevConvOutActive = torch.any(prevConvOutFlat > 0, dim=2)	#calculate if any activation for each channel	#shape=[B, C]
			b_idx, c_idx = torch.where(prevConvOutActive)				# both [N_active]
			xSubsetIndices = (b_idx, c_idx)				# tuple of two 1-D tensors
			#print("xSubsetIndices = ", xSubsetIndices)

			'''
			1. extract from each active batch channel (c_idx) a tensor of channel CNN kernel inputs called activeBatchChannelKernelInputs of shape = [numberOfActiveChannels, kernelInputW, kernelInputH, kernelWidth, kernelHeight]). len(c_idx) = numberOfActiveChannels. Each kernel input is of size kernelWidth x kernelHeight (e.g. 3x3) and is extracted with stride=1, so kernelInputW/kernelInputH is identical to the original image W/H.
			'''
			# ---------- inputs ----------
			# prevConvOut	  : [B, C, H, W]		  original activation maps
			# b_idx, c_idx	 : 1-D tensors from torch.where(mask) (# = N_active)
			# self.convKernels		  : [O, 1, kH, kW]		   static binary +1/-1 filters
			# ---------------------------------------

			# STEP 0 - gather only the active feature-maps
			x_active = prevConvOut[b_idx, c_idx]			# [N_active, H, W]
			#print("x_active.shape = ", x_active.shape)
		else:
			x_active = x	#shape 
		x_active = x_active.unsqueeze(1)				# [N_active, 1, H, W]
		#print("convLayerIndex = ", convLayerIndex, ", x_active.shape = ", x_active.shape)

		'''
		2. apply convOutChannels (eg 15) static binary (+1/-1) convolutional kernels (of shape = [convOutChannels, {1,} kernelInputH, kernelWidth]) to the activeBatchChannelKernelInputs. This can be done either by a) using some manual matrix operations of your choosing, or b) using a standard pytorch cnn kernel treating numberOfActiveChannels as the batch dimension and convInChannels=1 to a standard pytorch conv2d operation. Please implement at least method b (only implement method a if you think you have identified a faster way).
		'''
		kH, kW = self.convKernels.shape[-2:]

		# ---------------------------------------------------------------------------
		# 2) apply torch.conv2d
		# ---------------------------------------------------------------------------
		weight = self.convKernels	# [O, 1, 3, 3]
		O = weight.size(0)
		if(EISANICNNoptimisationAssumeInt8):
			x_active = x_active.float()
		#print("weight.shape = ", weight.shape)
		#print("x_active.sum() = ", x_active.sum())

		# ---------- overlap: how many 1-bits of the mask are present ----------
		overlap = F.conv2d(
			x_active.float(),          # [N_active,1,H,W]
			weight,                    # [O,1,3,3]
			stride=1,
			padding=1)                 # keep spatial size

		# ---------- number of 1s in each *mask* -------------------------------
		mask_sums = weight.view(O, -1).sum(dim=1)            # (O,)

		# ---------- number of 1s in each *patch* ------------------------------
		ones_kernel = weight.new_ones(1, 1, 3, 3)            # (1,1,3,3)
		patch_sum   = F.conv2d(x_active.float(), ones_kernel, stride=1, padding=1)  # [N_active,1,H,W]
		patch_sum   = patch_sum.repeat(1, O, 1, 1)           # broadcast to (N_active,O,H,W)

		mask_sum_broad = mask_sums.view(1, O, 1, 1)
		# ---------- exact-pattern match  (= AND =) ----------------------------
		# exact-pattern match (binary detector)
		convOut = ((overlap == mask_sum_broad) & (patch_sum == mask_sum_broad))
		if EISANICNNoptimisationAssumeInt8:
			convOut = convOut.to(torch.int8)
		else:
			convOut = convOut.float()

		#print("convOut.sum() = ", convOut.sum())

		# ---------------------------------------------------------------------------
		# 3) outputs
		# ---------------------------------------------------------------------------
		# convOut		: [N_active, O, H, W] final activations, 0 where needed
		# mask		   : [N_active, H, W]	 True = "kernel was applied"

		if(convLayerIndex == 0):
			x_active_new, C_new = postprocess_dense(self, convOut, b_idx, c_idx, C)
			b_idx_new = None
			c_idx_new = None
		else:
			x_active_new, b_idx_new, c_idx_new, C_new = postprocess_flat(self, convOut, b_idx, c_idx, C)

		return x_active_new, b_idx_new, c_idx_new, B, C_new


	# ------------------------------------------------------------------------------
	# 0)  sparse_to_dense() - restore shape  [B, C, H, W]   (zeros for every inactive channel)
	# ------------------------------------------------------------------------------

	def sparse_to_dense(self, x_active, b_idx, c_idx, B, C):
		"""
		x_active : [N_active*O, H, W]   - rows produced by postprocess_flat
		b_idx	: [N_active*O]		 - batch index per row
		c_idx	: [N_active*O]		 - channel index per row  (0 \u2026 C-1)
		B, C	 : ints - full tensor dims (C == C_old * O)

		returns dense_out : [B, C, H, W]  (zeros for every inactive slot)
		"""
		H, W = x_active.shape[-2:]

		# allocate a zero-filled buffer on the same device / dtype as the data
		dense_out = x_active.new_zeros(B, C, H, W)		# [B, C, H, W]

		# scatter each flat row back to its (batch, channel, h, w) slice
		dense_out[b_idx, c_idx] = x_active			# advanced indexing

		return dense_out


	# ------------------------------------------------------------------------------
	# 1)  postprocess_dense()  
	#converts directly back to original non-sparse image format (orig: not used as image channel data is stored as sparse during CNN phase for optimisation)
	# ------------------------------------------------------------------------------
	def postprocess_dense(self, convOut, b_idx, c_idx, B, C):

		H, W = convOut.shape[-2:]
		O = self.convKernels.shape[0]					# # output channels per input map

		# allocate an all-zero buffer on the right device / dtype
		full = torch.zeros((B, C, O, H, W), device=convOut.device)	# [B, C, O, H, W]

		# scatter: each (b_idx[k], c_idx[k]) slice receives convOut[k]
		full[b_idx, c_idx] = convOut					# advanced indexing, no loop

		# collapse C and O into one axis
		out = full.reshape(B, C * O, H, W)		# [B, C*O, H, W]

		C_new = C * O

		return out, C_new

	# ------------------------------------------------------------------------------
	# 1)  postprocess_flat()   <<[N_active, O, H, W]  -  [N_active*O, H, W]>>
	#	 - flattens the O-axis into the batch axis
	#	 - updates b_idx / c_idx for the new logical channel count = C*O
	# ------------------------------------------------------------------------------
	def postprocess_flat(self, convOut, b_idx, c_idx, C):
		"""
		convOut : [N_active, O, H, W]
		b_idx   : [N_active]		  batch ids of active inputs
		c_idx   : [N_active]		  channel ids  >>>>
		C	   : original C (needed to compute new channel numbers)

		returns x_active_new [N_active*O, H, W],
				b_idx_new	[N_active*O],
				c_idx_new	[N_active*O]  where channel axis = C*O
		"""
		N_active, O, H, W = convOut.shape

		# flatten O into the batch dim
		x_active_new = convOut.view(-1, H, W)					# [N_active*O, H, W]

		# repeat every (b,c) pair O times
		b_idx_new = b_idx.repeat_interleave(O)					# [N_active*O]

		# new channel id = c*O + o   (o  [0 ... O-1])
		o_range	= torch.arange(O, device=c_idx.device)		# [O]
		c_idx_new  = (c_idx.unsqueeze(1) * O + o_range).reshape(-1)

		C_new = C*O

		return x_active_new, b_idx_new, c_idx_new, C_new


	# ------------------------------------------------------------------------------
	# 3)  sparse_maxpool2d()   <<[N_active, H, W] - [N_active, H//2, W//2]>>
	#	 - 2x2 max-pool with stride 2 (change k,s if desired)
	# ------------------------------------------------------------------------------
	def sparse_maxpool2d(self, x, kernel_size=2, stride=2):
		"""
		x : [N_active, H, W]   (no channel axis)

		returns [N_active, H//stride, W//stride]
		"""
		return F.max_pool2d(x.unsqueeze(1).float(), kernel_size, stride).squeeze(1)



	def dense_conv(self, x: torch.Tensor) -> torch.Tensor:
		'''
		Vectorised mask-match:
		 - kernel bank shape ........ (K, 1, 3, 3)   (K = 511)
		 - input  ................... (B, C, H, W)
		 - output  .................. (B, C*K, H', W')

		A pixel fires (value 1) **iff** every 1 in the mask is present in the
		33 patch of the *same* input channel.
		'''
		B, C, H, W = x.shape
		device = x.device

		# ---------- kernel bank duplicated per input channel ----------
		kernels = self.convKernels.to(device).repeat(C, 1, 1, 1)      # (C*K,1,3,3)
		K = self.convKernels.size(0)
		if not hasattr(self, "_mask_sums"):
			self._mask_sums = self.convKernels.view(K, -1).sum(1)      # (K,)
		mask_sums = self._mask_sums.to(device).repeat(C)               # (C*K,)

		# ---------- overlap = how many mask-ones are present ----------
		overlap = F.conv2d(x.float(), kernels, stride=1, groups=C)     # (B,C*K,H',W')

		# ---------- patch_sum = how many ones in *entire* patch -------
		ones_kernel = torch.ones_like(self.convKernels[:1])            # (1,1,3,3)
		ones_kernel = ones_kernel.repeat(C, 1, 1, 1)                   # (C,1,3,3)
		patch_sum = F.conv2d(x.float(), ones_kernel, stride=1, groups=C)  # (B,C,H',W')
		patch_sum = patch_sum.repeat_interleave(K, dim=1)              # (B,C*K,H',W')

		# ---------- exact match : both equalities ---------------------
		match = ((overlap == mask_sums.view(1, -1, 1, 1)) &
				 (patch_sum == mask_sums.view(1, -1, 1, 1)))
		match = match.to(torch.int8 if EISANICNNoptimisationAssumeInt8 else torch.float32)

		B, Cnew, *spatial = match.shape
		return match, B, Cnew

elif(EISANICNNarchitectureDivergeLimitedKernelPermutations):

	def propagate_conv_layers_diverge_limited_kernel_permutations(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Full image pipeline:
		 - repeat {conv -> (activation) -> (optional max-pool every Q layers)} numberOfConvlayers times
		 - flatten to (batch, -1) int8
		Notes:
		 - Activation is applied *every* layer (right after conv).
		 - Max-pool runs only when (layerIndex+1) % Q == 0 (typical Q=2).
		"""
		z = x
		for convLayerIndex in range(self.config.numberOfConvlayers):
			# Valid 3x3, stride=1 grouped conv (reduces H,W by 2 per layer)
			z, B, C = conv(self, z)

			# Optional pointwise activation \u2014 executed EVERY layer
			if EISANICNNactivationFunction:
				z = apply_cnn_activation(self, z)

			# Max-pool only every Q layers (ceil_mode matches planner math)
			if CNNmaxPool and ((convLayerIndex + 1) % EISANICNNmaxPoolEveryQLayers == 0):
				z = F.max_pool2d(z.float(), kernel_size=2, stride=2, ceil_mode=True)

		linearInput = z.view(z.size(0), -1)  # (batch, encodedFeatureSize)
		return linearInput


	def conv(self, z: torch.Tensor):
		"""
		Apply the fixed/analytic kernel bank as a grouped convolution (VALID 3x3, stride=1).
		Input:
			z: (B, C_in, H, W)
		Returns:
			y: (B, C_out, H-2, W-2) with C_out = C_in * K, where K = self.convKernels.shape[0]
			B: batch size (int)
			C_out: number of output channels (int)
		"""
		assert z.dim() == 4, f"conv expects 4D tensor (B,C,H,W), got shape={tuple(z.shape)}"
		B, C_in, H, W = z.shape
		device = z.device
		# Kernel bank: (K,1,3,3), float, on same device
		kbank = self.convKernels.to(device).float()
		assert kbank.ndim == 4 and kbank.shape[1] == 1 and kbank.shape[2:] == (3, 3), f"self.convKernels must be (K,1,3,3); got {tuple(kbank.shape)}"
		K = kbank.shape[0]
		
		# Reshape so that each (batch, channel) pair becomes its own single-channel sample.
		# This allows us to reuse the kernel bank without repeating it ``C_in`` times and, more
		# importantly, makes it possible to chunk the convolution along this expanded batch axis
		# to stay within PyTorch's 32-bit indexing limits.
		z_contig = z.contiguous()
		BC = B * C_in
		z_reshaped = z_contig.view(BC, 1, H, W)
		
		if EISANICNNpaddingPolicy == 'same':
			H_out, W_out = H, W
		else:
			H_out = H - CNNkernelSize + 1
			W_out = W - CNNkernelSize + 1
		
		int32_max = 2**31 - 1
		# Determine how many (batch, channel) slices we can process per convolution call
		# without exceeding 32-bit indexing requirements for either the input or the output.
		max_slices_input = max(1, int32_max // (H * W))
		max_slices_output = max(1, int32_max // (K * H_out * W_out))
		chunk_size = min(BC, max_slices_input, max_slices_output)
		
		outputs = []
		for start in range(0, BC, chunk_size):
			end = min(start + chunk_size, BC)
			chunk = z_reshaped[start:end].to(torch.float32)
			if EISANICNNpaddingPolicy == 'same':
				if EISANICNNpaddingMode == 'zeros':
					y_chunk = F.conv2d(chunk, kbank, bias=None, stride=1, padding=1)
				elif EISANICNNpaddingMode == 'reflect':
					chunk = F.pad(chunk, (1, 1, 1, 1), mode=EISANICNNpaddingMode)
					y_chunk = F.conv2d(chunk, kbank, bias=None, stride=1, padding=0)
				else:
					printe(f"Unsupported EISANICNNpaddingMode: {EISANICNNpaddingMode}")
					y_chunk = F.conv2d(chunk, kbank, bias=None, stride=1, padding=1)
			else:
				y_chunk = F.conv2d(chunk, kbank, bias=None, stride=1, padding=0)
			outputs.append(y_chunk)
		
		y = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
		y = y.view(B, C_in, K, H_out, W_out)
		y = y.reshape(B, C_in * K, H_out, W_out)
		
		return y, B, y.shape[1]

	def apply_cnn_activation(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Configurable activation after conv, before pooling.
		self.config.CNNactivation E {'identity','relu','leaky_relu','gelu','abs','square','softshrink','threshold'}
		"""
		act = 'relu'
		if act == 'identity':
			return z
		elif act == 'relu':
			return F.relu(z)
		elif act == 'leaky_relu':
			neg = float(getattr(self.config, 'CNNleakySlope', 0.01))
			return F.leaky_relu(z, negative_slope=neg)
		elif act == 'gelu':
			return F.gelu(z)
		elif act == 'abs':			# full-wave rectification (phase-invariant magnitude)
			return z.abs()
		elif act == 'square':		# energy model; pairs well with pooling
			return z * z
		elif act == 'softshrink':	# sparse, signed shrinkage
			lmb = float(getattr(self.config, 'CNNsoftShrinkLambda', 0.05))
			return F.softshrink(z, lambd=lmb)
		elif act == 'threshold':	# signed hard-threshold: keep |z|>T, else 0
			T = float(getattr(self.config, 'CNNactivationThreshold', 0.0))
			return z * (z.abs() > T).to(z.dtype)
		else:
			return z

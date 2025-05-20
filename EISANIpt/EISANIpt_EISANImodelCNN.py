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

import torch
from torch import nn
from ANNpt_globalDefs import *
import itertools
import torch.nn.functional as F
import math 


# ---------------------------------------------------------
# IMAGE helpers (init & propagation)
# ---------------------------------------------------------

def _init_conv_layers(self) -> None:
	"""
	Initialise binary kernel bank (3  3, 1) and compute flattened
	output size after `numberOfConvlayers` (+ optional max-pooling).
	Sets:
	 self.convKernels     (512,1,3,3) int8
	 self.encodedFeatureSize
	"""
	assert CNNkernelSize == 3 and CNNstride == 1, "Only 33 stride-1 kernels supported in this implementation"

	# --- generate every possible 3  3 binary kernel (+1/-1)
	all_patterns = torch.tensor(list(itertools.product([-1, 1], repeat=CNNkernelSize*CNNkernelSize)), dtype=torch.int8)
	self.convKernels = all_patterns.view(512, 1, 3, 3)		# (outCh,inCh,H,W)

	# --- compute flattened size after N conv (+max-pool) layers
	H = inputImageHeight
	W = inputImageWidth
	ch = numberInputImageChannels
	print(f"conv0: channels={ch}, H={H}, W={W}")
	for layerIdx in range(self.config.numberOfConvlayers):
		H = H - CNNkernelSize + 1			# stride 1 conv
		W = W - CNNkernelSize + 1
		if CNNmaxPool:
			H //= 2
			W //= 2
		ch *= 512							# each conv multiplies channels by 512
		print(f"conv{layerIdx+1}: channels={ch}, H={H}, W={W}")
	self.encodedFeatureSize = ch * H * W

def _apply_all_binary_kernels(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Apply the pre-built 512-kernel bank to every channel separately,
	then threshold with `CNNkernelThreshold`.
	Returns new (batch, ch*512, H', W') tensor of int8 {0,1}.
	"""
	batch, inCh, H, W = x.shape
	out_list = []
	kernel_bank = self.convKernels.to(x.device).float()
	for c in range(inCh):
		conv_out = F.conv2d(x[:, c:c+1].float(), kernel_bank, stride=CNNstride)
		conv_bin = (conv_out >= CNNkernelThreshold).to(torch.int8)
		out_list.append(conv_bin)
	return torch.cat(out_list, dim=1)	# new channels = inCh*512

def _propagate_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Full image pipeline:
	 - threshold input channels at `EISANICNNinputChannelThreshold`
	 - repeat {conv -> (optional) max-pool} `numberOfConvlayers` times
	 - flatten to (batch, -1) int8
	"""
	z = (x >= EISANICNNinputChannelThreshold).to(torch.int8)
	for convLayerIndex in range(self.config.numberOfConvlayers):
		if EICNNoptimisationBlockwiseConv:
			z = conv_blockwise(z, self.convKernels.to(z.device).float(), 128)
		else:
			z = self._apply_all_binary_kernels(self, z)
		if CNNmaxPool:
			z = F.max_pool2d(z.float(), kernel_size=2, stride=2).to(torch.int8)
		if EICNNoptimisationPackBinary:
			z_packed, z_shape = pack_binary(z)
			z = unpack_binary(z_packed, z_shape)
	return z.view(z.size(0), -1)			# (batch, encodedFeatureSize)


def _torch_packbits_fallback(x: torch.Tensor, dim=-1) -> torch.Tensor:
	"""Pack binary {0,1} uint8/int8 along `dim` \u2192 uint8 tensor."""
	if x.dtype not in (torch.uint8, torch.int8):
		x = x.to(torch.uint8)
	if dim < 0:
		dim += x.ndim
	pad_len = (-x.size(dim)) % 8
	if pad_len:
		pad_shape = list(x.shape)
		pad_shape[dim] = pad_len
		x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
	x = x.view(*x.shape[:dim], -1, 8, *x.shape[dim+1:])
	shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
	x = (x << shifts).sum(dim=dim+1)
	return x

def _torch_unpackbits_fallback(x: torch.Tensor, dim=-1) -> torch.Tensor:
	"""Inverse of _torch_packbits_fallback."""
	if dim < 0:
		dim += x.ndim
	x = x.unsqueeze(dim+1)
	shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
	bits = (x >> shifts) & 1
	return bits.reshape(*x.shape[:dim], -1, *x.shape[dim+2:])

torch_has_pack = hasattr(torch, "packbits")

def pack_binary(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
	packed = (torch.packbits if torch_has_pack else _torch_packbits_fallback)(x.to(torch.uint8), dim=-1)
	return packed, x.shape

def unpack_binary(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
	"""Inverse of pack_binary, trimming pad bits to match `shape` exactly."""
	unpacked = (torch.unpackbits if torch_has_pack else _torch_unpackbits_fallback)(packed, dim=-1)
	n_needed = math.prod(shape)					   # total elements wanted
	unpacked = unpacked.flatten()[:n_needed]		  # drop padding zeros
	return unpacked.view(shape).to(torch.int8)
		
def conv_blockwise(x: torch.Tensor, kernels: torch.Tensor, block: int = 32) -> torch.Tensor:
	"""
	Apply `kernels` (shape [512,1,3,3]) to **each** input channel in blocks
	of size \u2264 `block`, keeping RAM low.
	Returns dense int8 (batch, inCh*512, H', W').
	"""
	batch, inCh, H, W = x.shape
	out = []
	for i in range(0, kernels.size(0), block):
		k_blk = kernels[i:i+block]                # (b,1,3,3)
		for c in range(inCh):
			y = F.conv2d(x[:, c:c+1].float(), k_blk, stride=1)
			y = (y >= CNNkernelThreshold).to(torch.int8)
			out.append(y)
	return torch.cat(out, dim=1)                 # concat over channel dim

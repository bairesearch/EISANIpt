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
	 self.convKernels	 (512,1,3,3) int8
	 self.encodedFeatureSize
	"""
	assert CNNkernelSize == 3 and CNNstride == 1, "Only 33 stride-1 kernels supported in this implementation"

	# --- generate every possible 3  3 binary kernel (+1/-1)
	all_patterns = torch.tensor(list(itertools.product([-1, 1], repeat=CNNkernelSize*CNNkernelSize)), dtype=torch.int8)
	self.convKernels = all_patterns.view(512, 1, 3, 3).float().to(device)		# (outCh,inCh,H,W)

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

def _propagate_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Full image pipeline:
	 - threshold input channels at `EISANICNNinputChannelThreshold`
	 - repeat {conv -> (optional) max-pool} `numberOfConvlayers` times
	 - flatten to (batch, -1) int8
	"""
	z = (x >= EISANICNNinputChannelThreshold)
	if(EICNNoptimisationAssumeInt8):
		z = z.to(torch.int8)
	else:
		z = z.float()
	b_idx, c_idx, B, C = (None, None, None, None)
	for convLayerIndex in range(self.config.numberOfConvlayers):
		if EICNNoptimisationSparseConv and convLayerIndex > 0:
			z, b_idx, c_idx, B, C = _sparse_conv(self, convLayerIndex, z, b_idx, c_idx, B, C)
		else:
			if EICNNoptimisationBlockwiseConv:
				z = conv_blockwise(z, self.convKernels, 128)
			else:
				z = _dense_conv(self, z)
		if CNNmaxPool:
			if EICNNoptimisationSparseConv and convLayerIndex > 0:
				z = sparse_maxpool2d(z, kernel_size=2, stride=2)
			else:
				z = F.max_pool2d(z.float(), kernel_size=2, stride=2)
				if(EICNNoptimisationAssumeInt8):
					z = z.to(torch.int8)
		if EICNNoptimisationPackBinary:
			z_packed, z_shape = pack_binary(z)
			z = unpack_binary(z_packed, z_shape)
	
	if EICNNoptimisationSparseConv:
		#linearInput = dense_flat_to_sparse(z, b_idx, c_idx, B, C)
		linearInput = dense_flat_to_dense(z, b_idx, c_idx, B, C)
	linearInput = z.view(z.size(0), -1)			# (batch, encodedFeatureSize)	#flatten for linear layers
	return linearInput
	
def _sparse_conv(self, convLayerIndex, x, b_idx, c_idx, B, C) -> torch.Tensor:

	'''
	select parts of the orig tensor x based on any result, returning xSubset, xSubsetIndices
	'''
		
	if(convLayerIndex == 0):
		print("_sparse_conv warning: convLayerIndex == 0, function designed for sparse input") 
	if(convLayerIndex <= 1):
		prevConvOut = x
		B, C, *spatial = prevConvOut.shape			# save spatial dims
		prevConvOutFlat = prevConvOut.view(B, C, -1)				# [B, C, W*H]	 #flatten width/height dim
		prevConvOutActive = torch.any(prevConvOutFlat > 0, dim=2)	#calculate if any activation for each channel	#shape=[B, C]
		b_idx, c_idx = torch.where(prevConvOutActive)				# both [N_active]
		#xSubsetIndices = (b_idx, c_idx)				# tuple of two 1-D tensors

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
		x_active = x_active.unsqueeze(1)				# [N_active, 1, H, W]
	else:
		x_active = x	#shape 
	
	'''
	2. apply convOutChannels (eg 15) static binary (+1/-1) convolutional kernels (of shape = [convOutChannels, {1,} kernelInputH, kernelWidth]) to the activeBatchChannelKernelInputs. This can be done either by a) using some manual matrix operations of your choosing, or b) using a standard pytorch cnn kernel treating numberOfActiveChannels as the batch dimension and convInChannels=1 to a standard pytorch conv2d operation. Please implement at least method b (only implement method a if you think you have identified a faster way).
	'''
	# ------------------------------------------------------------------------------
	# 1) extract every stride-1 kernel-sized patch
	#	result shape - [N_active, H, W, kH, kW]  (same H/W as input, no loops)
	# ------------------------------------------------------------------------------
	kH, kW = self.convKernels.shape[-2:]
	
	if(EICNNoptimisationAssumeInt8):														# [N_active, kH*kW, H*W]
		patches = unfold_int8_stride1(x_active, CNNkernelSize)	#F.unfold not implemented for int8 (char)
	else:
		patches = F.unfold(
			x_active,												# [N_active, 1, H, W]
			kernel_size=(kH, kW),
			padding=(kH // 2, kW // 2),		# "same" so output H_out = H, W_out = W
			stride=1
		)
	
	# Method C (torch.conv2d optimised): 
	# mask per position: True if any element of the 3x3 window > 0
	mask_flat = patches.gt(0).any(dim=1)	# [N_active, H*W]
	mask	 = mask_flat.view(x_active.shape[0], *x_active.shape[2:])	# [N_active, H, W]

	# ---------------------------------------------------------------------------
	# 2) fast path -- use torch.conv2d as before, then apply the mask
	# ---------------------------------------------------------------------------
	weight = self.convKernels	# [O, 1, 3, 3]

	convOut = F.conv2d(
		x_active,									# [N_active, 1, H, W]
		weight,										# [O, 1, 3, 3]
		stride=1,
		padding=1									# kH // 2
	)												# [N_active, O, H, W]
	
	# zero-out positions that failed the "any>0" test
	convOut *= mask.unsqueeze(1)					# broadcast to [N_active, O, H, W]

	# ---------------------------------------------------------------------------
	# 3) outputs
	# ---------------------------------------------------------------------------
	# convOut		: [N_active, O, H, W] final activations, 0 where needed
	# mask		   : [N_active, H, W]	 True = "kernel was applied"
	
	x_active_new, b_idx_new, c_idx_new, C_new = postprocess_flat(convOut, b_idx, c_idx, C)
	
	return x_active_new, b_idx_new, c_idx_new, B, C_new

	
# ------------------------------------------------------------------------------
# 0)  dense_flat_to_dense() - restore shape  [B, C, H, W]   (zeros for every inactive channel)
# ------------------------------------------------------------------------------

def dense_flat_to_dense(x_active, b_idx, c_idx, B, C):
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


#converts directly back to original non-sparse image format (orig: not used as image channel data is stored as sparse during CNN phase for optimisation)
def dense_old_to_dense_new(convOut, b_idx, c_idx, B, C):

	H, W = convOut.shape[-2:]
	O = self.convKernels.shape[0]					# # output channels per input map

	# allocate an all-zero buffer on the right device / dtype
	full = torch.zeros((B, C, O, H, W), device=convOut.device)	# [B, C, O, H, W]

	# scatter: each (b_idx[k], c_idx[k]) slice receives convOut[k]
	full[b_idx, c_idx] = convOut					# advanced indexing, no loop

	# collapse C and O into one axis
	out = full.reshape(B, C * O, H, W)		# [B, C*O, H, W]

	C_new = C * O

	return out
	
# ------------------------------------------------------------------------------
# 1)  postprocess_flat()   <<[N_active, O, H, W]  -  [N_active*O, H, W]>>
#	 - flattens the O-axis into the batch axis
#	 - updates b_idx / c_idx for the new logical channel count = C*O
# ------------------------------------------------------------------------------
def postprocess_flat(convOut, b_idx, c_idx, C):
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
# 2)  dense_flat_to_sparse()   <<[N_active*O, H, W] - sparse COO [B, C, H, W]>>
#	 - builds the COO tensor directly, no gigantic dense buffer
# ------------------------------------------------------------------------------
def dense_flat_to_sparse(x_active, b_idx, c_idx, B, C):
	"""
	x_active : [N_active*O, H, W]
	b_idx	: batch index per row   (from postprocess_flat)
	c_idx	: channel index per row (from postprocess_flat)
	B, C	   : full tensor dims

	returns torch.sparse_coo_tensor of shape [B, C, H, W]
	"""
	H, W = x_active.shape[1:]
	nz   = torch.nonzero(x_active, as_tuple=False)		# [nnz, 3]

	row, h_idx, w_idx = nz.t()								# 1-D each

	indices = torch.stack((
		b_idx[row],			# batch dim
		c_idx[row],			# channel dim
		h_idx,					# height
		w_idx					# width
	), dim=0)												# [4, nnz]

	values  = x_active[row, h_idx, w_idx]				# [nnz]

	return torch.sparse_coo_tensor(
		indices, values,
		size=(B, C, H, W),
		dtype=x_active.dtype,
		device=x_active.device
	)


# ------------------------------------------------------------------------------
# 3)  sparse_maxpool2d()   <<[N_active, H, W] - [N_active, H//2, W//2]>>
#	 - 2x2 max-pool with stride 2 (change k,s if desired)
# ------------------------------------------------------------------------------
def sparse_maxpool2d(x, kernel_size=2, stride=2):
	"""
	x : [N_active, H, W]   (no channel axis)

	returns [N_active, H//stride, W//stride]
	"""
	return F.max_pool2d(x.unsqueeze(1), kernel_size, stride).squeeze(1)


'''
def _dense_conv(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Apply the pre-built 512-kernel bank to every channel separately,
	then threshold with `CNNkernelThreshold`.
	Returns new (batch, ch*512, H', W') tensor of int8 {0,1}.
	"""
	batch, inCh, H, W = x.shape
	out_list = []
	kernel_bank = self.convKernels
	for c in range(inCh):
		cInput = x[:, c:c+1]
		if(EICNNoptimisationAssumeInt8):
			cInput = cInput.float()
		conv_out = F.conv2d(x[:, c:c+1].float(), kernel_bank, stride=CNNstride)
		conv_bin = (conv_out >= CNNkernelThreshold)
		if(EICNNoptimisationAssumeInt8):
			conv_bin = conv_bin.to(torch.int8)
		else:
			conv_bin = conv_bin.float()
		out_list.append(conv_bin)
	return torch.cat(out_list, dim=1)	# new channels = inCh*512
'''

def _dense_conv(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Vectorized version: applies the same 512 kernels independently to each input channel
	in one grouped convolution, then thresholds.
	"""
	B, inCh, H, W = x.shape
	kH, kW = CNNkernelSize, CNNkernelSize		   # e.g. 3, 3
	# assume self.convKernels is [512, 1, kH, kW]
	bank = self.convKernels		 # [512,1,kH,kW]
	# Tile that bank in the out-channel dimension:
	#   \u2192 [inCh*512, 1, kH, kW]
	bank_rep = bank.repeat(inCh, 1, 1, 1)
	cInput = x
	if(EICNNoptimisationAssumeInt8):
		cInput = cInput.float()
	conv_out = F.conv2d(
		cInput,										 # [B, inCh, H, W]
		weight=bank_rep,								   # [inCh*512, 1, kH, kW]
		stride=CNNstride,
		groups=inCh										# split both input & weight
	)
	# conv_out is now [B, inCh*512, H', W']
	out = (conv_out >= CNNkernelThreshold)
	if(EICNNoptimisationAssumeInt8):
		out = out.to(torch.int8)
	else:
		out = out.float()
	return out

if(EICNNoptimisationAssumeInt8):
		
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
			k_blk = kernels[i:i+block]				# (b,1,3,3)
			for c in range(inCh):
				y = F.conv2d(x[:, c:c+1].float(), k_blk, stride=1)
				y = (y >= CNNkernelThreshold).to(torch.int8)
				out.append(y)
		return torch.cat(out, dim=1)				 # concat over channel dim

	def unfold_int8_stride1(x, k=3):
		"""
		x : [N_active, 1, H, W]  (int8 or any dtype)
		k : kernel size (odd, e.g. 3)

		returns patches : [N_active, k*k, H*W]  (view, no copy)
		"""
		N, _, H, W = x.shape
		if N == 0:											# nothing active \u2192 nothing to do
			return x.new_empty((0, k * k, H * W))

		p = k // 2
		xp = F.pad(x, (p, p, p, p))						# SAME pad, still int8

		# unfold height then width \u2192 (N, 1, H, W, k, k)
		patch_view = xp.unfold(2, k, 1).unfold(3, k, 1)

		# reshape explicitly: (N, k*k, H, W)
		patch_view = patch_view.reshape(N, k * k, H, W)

		# final flatten to (N, k*k, H*W)
		return patch_view.reshape(N, k * k, H * W)





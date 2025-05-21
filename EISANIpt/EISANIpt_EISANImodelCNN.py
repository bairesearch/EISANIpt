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
	 self.convKernels	 (511,1,3,3) binary +1/0 (stored as float for conv2d)
	 self.encodedFeatureSize
	"""
	assert CNNkernelSize == 3 and CNNstride == 1, "Only 33 stride-1 kernels supported in this implementation"

	# --- generate every possible 3 × 3 BINARY mask (+1 / 0)
	all_patterns = torch.tensor(list(itertools.product([0, 1], repeat=CNNkernelSize * CNNkernelSize)), dtype=torch.uint8)
	all_patterns = all_patterns[(all_patterns.sum(dim=1) > 0)]	# drop the all-zero mask (would match everywhere)
	self.convKernels = all_patterns.view(-1, 1, 3, 3).float().to(device)		# (outCh,inCh,H,W)	# (2**9)-1=511 out-channel
	self.convKernels = self.convKernels.float()	#torch.conv2d currently requires float
	print("self.convKernels.shape = ", self.convKernels.shape)
	
	# --- compute flattened size after N conv (+max-pool) layers
	H = inputImageHeight
	W = inputImageWidth
	ch = numberInputImageChannels
	print(f"conv0: channels={ch}, H={H}, W={W}")
	for layerIdx in range(self.config.numberOfConvlayers):
		H = H - CNNkernelSize + 1			# stride 1 conv
		W = W - CNNkernelSize + 1
		if CNNmaxPool:
			# use *ceil* division (same as F.max_pool2d(..., ceil_mode=True))
			H = (H + 1) // 2	#orig: H //= 2
			W = (W + 1) // 2	#orig: W //= 2
		ch *= (2**9)-1							# each conv multiplies channels by 511
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
				z, B, C = _dense_conv(self, z)
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
	
	if EICNNoptimisationSparseConv and self.config.numberOfConvlayers>1:
		#linearInput = dense_flat_to_sparse(z, b_idx, c_idx, B, C)
		linearInput = dense_flat_to_dense(z, b_idx, c_idx, B, C)
	else:
		linearInput = z
	linearInput = linearInput.view(linearInput.size(0), -1)			# (batch, encodedFeatureSize)	#flatten for linear layers
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
	print("convLayerIndex = ", convLayerIndex, ", x_active.shape = ", x_active.shape)

	'''
	2. apply convOutChannels (eg 15) static binary (+1/-1) convolutional kernels (of shape = [convOutChannels, {1,} kernelInputH, kernelWidth]) to the activeBatchChannelKernelInputs. This can be done either by a) using some manual matrix operations of your choosing, or b) using a standard pytorch cnn kernel treating numberOfActiveChannels as the batch dimension and convInChannels=1 to a standard pytorch conv2d operation. Please implement at least method b (only implement method a if you think you have identified a faster way).
	'''
	kH, kW = self.convKernels.shape[-2:]
	
	# ---------------------------------------------------------------------------
	# 2) apply torch.conv2d
	# ---------------------------------------------------------------------------
	weight = self.convKernels	# [O, 1, 3, 3]
	O = weight.size(0)
	if(EICNNoptimisationAssumeInt8):
		x_active = x_active.float()
	#print("weight.shape = ", weight.shape)
	print("x_active.sum() = ", x_active.sum())
	
	# ---------- overlap count (same as ordinary conv) ----------
	overlap = F.conv2d(                                   # (N_active, O, H, W)
    	x_active,                                 # [N_active, 1, H, W]  \u2013 0/1
    	weight,                                          # [O, 1, 3, 3]         \u2013 0/1
    	stride=1,
    	padding=1                                        # 3//2
	)

	# ---------- number of ones in each mask ---------------
	# pre-compute once and cache if you like
	mask_sums = weight.view(O, -1).sum(dim=1)            # (O,)

	# ---------- full-mask match ---------------------------
	convOut = (overlap == mask_sums.view(1, O, 1, 1))
	# convOut shape: [N_active, O, H, W], values {0,1}

	if(EICNNoptimisationAssumeInt8):
		convOut = convOut.to(torch.int8)
	else:
		convOut = convOut.float()
	
	print("convOut.sum() = ", convOut.sum())
	
	# ---------------------------------------------------------------------------
	# 3) outputs
	# ---------------------------------------------------------------------------
	# convOut		: [N_active, O, H, W] final activations, 0 where needed
	# mask		   : [N_active, H, W]	 True = "kernel was applied"
	
	if(convLayerIndex == 0):
		x_active_new, C_new = postprocess_dense(convOut, b_idx, c_idx, C)
		b_idx_new = None
		c_idx_new = None
	else:
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


# ------------------------------------------------------------------------------
# 1)  postprocess_dense()  
#converts directly back to original non-sparse image format (orig: not used as image channel data is stored as sparse during CNN phase for optimisation)
# ------------------------------------------------------------------------------
def postprocess_dense(convOut, b_idx, c_idx, B, C):

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
	return F.max_pool2d(x.unsqueeze(1).float(), kernel_size, stride).squeeze(1)



def _dense_conv(self, x: torch.Tensor) -> torch.Tensor:
	'''
	Vectorised mask-match:
	 - kernel bank shape ........ (K, 1, 3, 3)   (K = 511)
	 - input  ................... (B, C, H, W)
	 - output  .................. (B, C*K, H', W')
	
	A pixel fires (value 1) **iff** every 1 in the mask is present in the
	33 patch of the *same* input channel.
	'''
	B, C, H, W = x.shape
	device      = x.device
	
	#----- prepare kernel bank for grouped conv ---------------------------
	# Repeat the 511 masks once for each input channel.
	kernels = self.convKernels.to(device).repeat(C, 1, 1, 1)   # (C*K, 1, 3, 3)
	
	# Number of 1-bits in each mask (repeated per channel).
	if not hasattr(self, "_mask_sums"):
		self._mask_sums = self.convKernels.view(self.convKernels.size(0), -1).sum(1)  # (K,)
	mask_sums = self._mask_sums.to(device).repeat(C)                                   # (C*K,)
	
	#----- grouped convolution -------------------------------------------
	# groups=C -> each group uses its own slice of C_out=C*K channels
	if(EICNNoptimisationAssumeInt8):
		cInput = x.float()
	else:
		cInput = x
	overlap = F.conv2d(cInput, kernels, stride=1, groups=C)        # (B, C*K, H', W')
	
	#----- full-mask match ------------------------------------------------
	match = (overlap == mask_sums.view(1, -1, 1, 1))   # (B, C*K, H', W')
	if(EICNNoptimisationAssumeInt8):
		match = match.to(torch.int8)
	else:
		match = match.float()
		
	B, Cnew, *spatial = match.shape
		
	return match, B, Cnew


'''
def _dense_conv(self, x: torch.Tensor) -> torch.Tensor:
	"""
	Mask-match: output 1 only when **all** 1-positions in the mask are
	present in the input patch.
	"""
	batch, inCh, H, W = x.shape
	out = []
	kernel_bank = self.convKernels.to(x.device)               # float32, 0/1
	mask_sums   = kernel_bank.view(kernel_bank.size(0), -1).sum(dim=1)  # (511,)

	for c in range(inCh):
		# count how many 1s overlap between input and mask
		cInput = x[:, c:c+1]
		if(EICNNoptimisationAssumeInt8):
			cInput = cInput.float()
		overlap = F.conv2d(cInput, kernel_bank, stride=1)   # (B,511,H',W')
		# full match when overlap == number of 1s in that mask
		match = (overlap == mask_sums.view(1, -1, 1, 1))
		if(EICNNoptimisationAssumeInt8):
			match = match.to(torch.int8)
		else:
			match = match.float()

		out.append(match)


	outAll = torch.cat(out, dim=1)    # (B, inCh*511, H', W')
	
	B, Cnew, *spatial = outAll.shape

	return outAll, B, Cnew
'''
 

 

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
		Apply `kernels` (shape [511,1,3,3]) to **each** input channel in blocks
		of size \u2264 `block`, keeping RAM low.
		Returns dense int8 (batch, inCh*511, H', W').
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





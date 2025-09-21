"""EISANIpt_EISANImodelContinuousVarEncoding.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt model Continuous Var Encoding

"""

import torch
from ANNpt_globalDefs import *
import torch.nn.functional as F
if(useNLPDataset):
	import EISANIpt_EISANImodelNLP
elif(useImageDataset):
	import EISANIpt_EISANImodelCNN
	
# -------------------------------------------------------------
# Continuous var encoding as bits
# -------------------------------------------------------------

def continuousVarEncoding(self, x):
	if useTabularDataset:
		encoded = encodeContinuousVarsAsBitsWrapper(self, x)
		initActivation = encoded.to(torch.int8)
		numSubsamples = 1
	elif useImageDataset:
		if(EISANICNNuseBinaryInput):
			encoded = encodeContinuousVarsAsBitsWrapper(self, x)
			initActivation = EISANIpt_EISANImodelCNN.propagate_conv_layers_binary(self, encoded)	# (batch, encodedFeatureSize)
		else:
			initActivation = EISANIpt_EISANImodelCNN.propagate_conv_layers_float(self, x)	# (batch, encodedFeatureSize) int8
			if(EISANITABcontinuousVarEncodingNumBitsAfterCNN == 1):
				# Binarize/threshold channels for the linear stage, then flatten
				initActivation = (initActivation > EISANICNNinputChannelThreshold)
			else:
				initActivation = encodeContinuousVarsAsBits(self, initActivation, "useTabularDataset", useContinuousVarEncodeMethodAfterCNN, EISANITABcontinuousVarEncodingNumBitsAfterCNN, useVectorisedImplementation=True) 
			initActivation = initActivation.to(torch.int8)	# (batch, encodedFeatureSize) int8
	elif useNLPDataset:
		#print("x.shape = ", x.shape)
		#print("x = ", x)
		embedding = EISANIpt_EISANImodelNLP.encodeTokensInputIDs(self, x)	# (batch, sequenceLength, embeddingSize) float32
		#print("embedding.shape = ", embedding.shape)
		encoded = encodeContinuousVarsAsBitsWrapper(self, embedding)	#int8	#[batchSize, sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
		#print("encoded = ", encoded)
		initActivation = encoded.to(torch.int8)
		#encodedFeatureSize = EISANIpt_EISANImodelNLP.getEncodedFeatureSize()	#sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
	return initActivation

def encodeContinuousVarsAsBitsWrapper(self, x: torch.Tensor) -> torch.Tensor:
	#!useVectorisedImplementation is required for encodeDatasetBoolValuesAs1Bit
	return encodeContinuousVarsAsBits(self, x, datasetType, useContinuousVarEncodeMethod, EISANIcontinuousVarEncodingNumBits, useVectorisedImplementation=False)	
		
def encodeContinuousVarsAsBits(self, x: torch.Tensor, encodeDataset, encodeMethod, numBits, useVectorisedImplementation) -> torch.Tensor:
	if(encodeDataset=="useTabularDataset"):
		B, L = x.shape
	if(encodeDataset=="useImageDataset"):
		if numBits == 1:
			return x
		B, C, H, W = x.shape
		x = x.view(B, C * H * W)  # Flatten pixel dimensions
	elif(encodeDataset=="useNLPDataset"):
		if(useTokenEmbedding):
			B, L, E = x.shape
			x = x.view(B, L * E)
		else:
			B, L = x.shape
			x = x.view(B, L)

	if useVectorisedImplementation:
		if encodeMethod=="directBinary":
			encoded_bits = binary_code_encode_vectorised(self, x, numBits, continuousVarMin, continuousVarMax)
		elif encodeMethod=="grayCode":
			encoded_bits = gray_code_encode_vectorised(self, x, numBits, continuousVarMin, continuousVarMax)
		elif encodeMethod=="thermometer":
			encoded_bits = thermometer_encode_vectorised(self, x, numBits, continuousVarMin, continuousVarMax)
		elif encodeMethod=="onehot":
			encoded_bits = F.one_hot(x, num_classes=numBits)	# (B, L, numBits)
		else:
			printe("invalid encodeMethod: ", encodeMethod)
	else:
		if encodeMethod=="directBinary":
			encoded_bits_list = binary_code_encode(self, x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		elif encodeMethod=="grayCode":
			encoded_bits_list = gray_code_encode(self, x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		elif encodeMethod=="thermometer":
			encoded_bits_list = thermometer_encode(self, x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
		elif encodeMethod=="onehot":
			encoded_bits = F.one_hot(x, num_classes=numBits)	# (B, L, numBits)
		else:
			printe("invalid encodeMethod: ", encodeMethod)
				
	if(encodeDataset=="useTabularDataset"):
		if useVectorisedImplementation:
			code = encoded_bits.reshape(B, -1)	# encoded_bits: [B, nCont, numBits] -> [B, nCont*numBits]
		else:
			code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension	#[B, nCont*numBits]
	elif(encodeDataset=="useImageDataset"):
		if useVectorisedImplementation:
			pass	#already [B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
		else:
			code = torch.stack(encoded_bits_list, dim=2) 	#[B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
		code = code.view(B, C, H, W, numBits)	#unflatten pixel dimensions
		code = code.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, C, numBits, H, W]
		code = code.reshape(B, C*numBits, H, W)
	elif(encodeDataset=="useNLPDataset"):
		if encodeMethod=="onehot":
			code = encoded_bits.view(B, L, numBits) 	#[B, L*numBits]
			#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, EISANINLPcontinuousVarEncodingNumBits]
			code = code.reshape(B, L*numBits)
		else:
			if useVectorisedImplementation:
				code = encoded_bits	#[B, L, numBits]
			else:
				code = torch.stack(encoded_bits_list, dim=1) 	#[B, L, numBits]
			#code = code.reshape(B, L, E*numBits)		#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
			code = code.reshape(B, L*E*numBits)

	return code

def continuous_to_int_vectorised(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""
	Map (B, L) continuous tensor to integer levels in [0, 2**numBits - 1].
	Robust to minVal == maxVal.
	"""
	if numBits <= 0:
		raise ValueError("numBits must be >= 1")
	x = x if x.is_floating_point() else x.to(torch.float32)
	xClamped = torch.clamp(x, minVal, maxVal)
	denom = max(maxVal - minVal, torch.finfo(xClamped.dtype).tiny)
	norm = (xClamped - minVal) / denom
	scaled = norm * ((1 << numBits) - 1)
	return scaled.round().to(torch.long)


def binary_code_encode_vectorised(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""
	Direct binary (base-2) encoding. Returns bits as LSB-first.
	Shape: (B, L, numBits), values {0.0, 1.0}.
	"""
	intLevels = continuous_to_int_vectorised(self, x, numBits, minVal, maxVal)  # (B, L), ints in [0, 2**numBits-1]
	bitPositions = torch.arange(numBits, device=x.device, dtype=torch.long)  # [0..numBits-1], LSB-first
	bits = ((intLevels.unsqueeze(-1) >> bitPositions) & 1).to(x.dtype)
	return bits


def gray_code_encode_vectorised(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""
	Vectorized Gray-code encoding. Returns bits as LSB-first.
	Shape: (B, L, numBits), values {0.0, 1.0}.
	"""
	intLevels = continuous_to_int_vectorised(self, x, numBits, minVal, maxVal)  # (B, L)
	grayLevels = intLevels ^ (intLevels >> 1)                   # (B, L)
	bitPositions = torch.arange(numBits, device=x.device, dtype=torch.long)
	bits = ((grayLevels.unsqueeze(-1) >> bitPositions) & 1).to(x.dtype)
	return bits


def thermometer_encode_vectorised(x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""
	Vectorized thermometer encoding with exactly `numBits` thresholds.
	Encodes into LSB-first order where bit k indicates level >= k.
	Shape: (B, L, numBits), values {0.0, 1.0}.

	Note: Thermometer encoding uses `numBits` discrete levels (0..numBits-1),
	not 2**numBits levels like binary/Gray.
	"""
	if numBits <= 0:
		raise ValueError("numBits must be >= 1")
	xf = x if x.is_floating_point() else x.to(torch.float32)
	xClamped = torch.clamp(xf, minVal, maxVal)
	denom = max(maxVal - minVal, torch.finfo(xClamped.dtype).tiny)
	norm = (xClamped - minVal) / denom  # [0,1]
	levels = (norm * (numBits - 1)).round().to(torch.long)  # (B, L), in [0, numBits-1]

	thresholds = torch.arange(numBits, device=x.device, dtype=torch.long)  # 0..numBits-1
	bits = (levels.unsqueeze(-1) >= thresholds).to(x.dtype)  # (B, L, numBits)
	return bits



def binary_code_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
	"""Direct binary encoding (batch, features, bits).

	Maps continuous values to integer levels then bit-slices directly. Boolean features pass-through as 1-bit.
	Designed to provide, on average, balanced 0/1 per bit for uniformly distributed inputs.
	"""
	batch_size, num_features = x.shape
	device = x.device

	# Identify Boolean columns
	isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
	if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
		limit = min(num_features, len(fieldTypeList))
		isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

	# Boolean bits
	boolBits = x[:, isBool].float()

	# Continuous -> integer levels -> bit-slice
	if (~isBool).any():
		xCont = x[:, ~isBool]
		intLevels = continuous_to_int(self, xCont, numBits, minVal, maxVal)  # (batch, nCont)
		bitPos = torch.arange(numBits, device=device)
		contBits = ((intLevels.unsqueeze(-1) >> bitPos) & 1).float()  # (batch, nCont, numBits)
	else:
		contBits = x.new_empty(batch_size, 0, numBits)

	# Interleave back
	encoded_bits_list: list[torch.Tensor] = []
	bool_iter = iter(boolBits.unbind(dim=1))
	cont_iter = iter(contBits.unbind(dim=1))
	for flag in isBool:
		encoded_bits_list.append(next(bool_iter).unsqueeze(1) if flag else next(cont_iter))	
	return encoded_bits_list

def gray_code_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
	"""Vectorised Gray-code encoding (batch, features, bits) with no costly Python loop."""
	batch_size, num_features = x.shape
	device = x.device

	# 1. Identify Boolean columns once
	isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
	if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
		limit = min(num_features, len(fieldTypeList))
		isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

	# 2. Boolean bits (1-bit, already 0/1)
	boolBits = x[:, isBool].float()											# (batch, nBool)

	# 3. Continuous bits - quantise - Gray - bit-slice
	if (~isBool).any():
		xCont = x[:, ~isBool]												# (batch, nCont)
		intLevels = continuous_to_int(self, xCont, numBits, minVal, maxVal)		# (batch, nCont)
		grayLevels = intLevels ^ (intLevels >> 1)
		bitPos = torch.arange(numBits, device=device)						# (numBits,)
		contBits = ((grayLevels.unsqueeze(-1) >> bitPos) & 1).float()		# (batch, nCont, numBits)
	else:
		contBits = x.new_empty(batch_size, 0, numBits)

	# 4. Re-interleave (cheap Python, negligible cost)
	encoded_bits_list: list[torch.Tensor] = []
	bool_iter = iter(boolBits.unbind(dim=1))									# each - (batch,)
	cont_iter = iter(contBits.unbind(dim=1))								# each - (batch, numBits)
	for flag in isBool:														# trivial loop
		encoded_bits_list.append(next(bool_iter).unsqueeze(1) if flag else next(cont_iter))
	return encoded_bits_list

def thermometer_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
	"""Vectorised thermometer encoding (batch, features, bits) without per-feature Python work."""
	batch_size, num_features = x.shape
	device = x.device

	isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
	if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
		limit = min(num_features, len(fieldTypeList))
		isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

	boolBits = x[:, isBool].float()											# (batch, nBool)

	if (~isBool).any():
		xCont = x[:, ~isBool]												# (batch, nCont)
		intLevels = continuous_to_int(self, xCont, numBits, minVal, maxVal)		# (batch, nCont)
		thresholds = torch.arange(numBits, device=device)					# (numBits,)
		contBits = (intLevels.unsqueeze(-1) >= thresholds).float()			# (batch, nCont, numBits)
	else:
		contBits = x.new_empty(batch_size, 0, numBits)

	encoded_bits_list: list[torch.Tensor] = []
	bool_iter = iter(boolBits.unbind(dim=1))
	cont_iter = iter(contBits.unbind(dim=1))
	for flag in isBool:
		encoded_bits_list.append(next(bool_iter).unsqueeze(1) if flag else next(cont_iter))
	return encoded_bits_list

def continuous_to_int(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""Map (batch, features) continuous tensor to integer levels [0, 2**bits-1]."""
	xClamped = torch.clamp(x, minVal, maxVal)
	norm = (xClamped - minVal) / (maxVal - minVal)
	scaled = norm * (2 ** numBits - 1)
	return scaled.round().to(torch.long)

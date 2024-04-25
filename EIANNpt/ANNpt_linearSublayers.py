"""ANNpt_linearSublayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt linear sublayers

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *

class LinearSegregated(nn.Module):
	def __init__(self, in_features, out_features, number_sublayers):
		super().__init__()
		if(useCNNlayers):
			self.segregatedLinear = nn.Conv2d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=CNNkernelSize, stride=CNNstride, padding=CNNpadding, groups=number_sublayers)
		else:	
			self.segregatedLinear = nn.Conv1d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=1, groups=number_sublayers)
		self.number_sublayers = number_sublayers
		
	def forward(self, x):
		#x.shape = batch_size, number_sublayers, in_features
		if(useCNNlayers):
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
		else:
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], 1)
		x = self.segregatedLinear(x)
		if(useCNNlayers):
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers, x.shape[2], x.shape[3])
		else:
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers)
		#x.shape = batch_size, number_sublayers, out_features
		return x

def generateLinearLayer(self, layerIndex, config, parallelStreams=False, sign=True, inFeaturesMatchHidden=False, inFeaturesMatchOutput=False):
	if(inFeaturesMatchHidden):
		if(layerIndex == 0):
			in_features = config.hiddenLayerSize
			out_features = config.hiddenLayerSize
		else:
			printe("generateLinearLayer error: inFeaturesMatchHidden and layerIndex != 0")	
	elif(inFeaturesMatchOutput):
		if(layerIndex == config.numberOfLayers-1):
			in_features = config.outputLayerSize
			out_features = config.outputLayerSize
		else:
			printe("generateLinearLayer error: outFeaturesMatchInput and layerIndex != config.numberOfLayers-1")
	else:
		if(inputLayerInList and layerIndex == 0):
			in_features = config.inputLayerSize
		else:
			in_features = config.hiddenLayerSize
		if(outputLayerInList and layerIndex == config.numberOfLayers-1):
			out_features = config.outputLayerSize
		else:
			out_features = config.hiddenLayerSize
	linearSublayersNumber = config.linearSublayersNumber
	return generateLinearLayer2(self, layerIndex, in_features, out_features, linearSublayersNumber, parallelStreams, sign)
		
def generateLinearLayer2(self, layerIndex, in_features, out_features, linearSublayersNumber, parallelStreams=False, sign=True):
	if(getUseLinearSublayers(self, layerIndex)):
		linear = LinearSegregated(in_features=in_features, out_features=out_features, number_sublayers=linearSublayersNumber)
	else:
		if(useCNNlayers):
			linear = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=CNNkernelSize, stride=CNNstride, padding=CNNpadding)
		else:
			if(parallelStreams):
				in_features = in_features*linearSublayersNumber
			linear = nn.Linear(in_features=in_features, out_features=out_features)

	weightsSetLayer(self, layerIndex, linear, sign)

	return linear

def generateActivationFunction():
	if(activationFunctionType=="softmax"):
		if(thresholdActivations):
			activation = OffsetSoftmax(thresholdActivationsMin)
		else:
			activation = nn.Softmax(dim=1)
	elif(activationFunctionType=="relu"):
		if(thresholdActivations):
			activation = OffsetReLU(thresholdActivationsMin)
		else:
			activation = nn.ReLU()
	elif(activationFunctionType=="none"):
		activation = None
	return activation

def generateActivationLayer(self, layerIndex, config):
	return generateActivationFunction()

def executeLinearLayer(self, layerIndex, x, linear, parallelStreams=False, sign=True):
	if(useSignedWeights):
		weightsFixLayer(self, layerIndex, linear, sign)	#otherwise need to constrain backprop weight update function to never set weights below 0
	if(getUseLinearSublayers(self, layerIndex)):
		#perform computation for each sublayer independently
		if(not parallelStreams):
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		x = linear(x)
	else:
		if(parallelStreams):
			x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
			#print("x.shape = ", x.shape)
		x = linear(x)
	return x

def executeActivationLayer(self, layerIndex, x, activationFunction, parallelStreams=False, executeActivationFunctionOverFeatures=True):
	if(normaliseActivationSparsity):
		x = nn.functional.layer_norm(x, x.shape[1:])   #normalized_shape does not include batchSize
	if(getUseLinearSublayers(self, layerIndex) and not simulatedDendriticBranches):
		if(activationFunctionType=="softmax"):
			if(executeActivationFunctionOverFeatures):
				numberOfSamples = x.shape[0]
				numberOfSublayers = x.shape[1]
				if(useCNNlayers):
					x = pt.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
				else:
					x = pt.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
			x = activationFunction(x)
			if(executeActivationFunctionOverFeatures):
				if(useCNNlayers):
					x = pt.reshape(x, (numberOfSamples, numberOfSublayers, x.shape[1], x.shape[2], x.shape[3]))
				else:
					x = pt.reshape(x, (numberOfSamples, numberOfSublayers, x.shape[1]))
		elif(activationFunctionType=="relu"):
			x = activationFunction(x)
		elif(activationFunctionType=="none"):
			pass
		if(not parallelStreams):
			if(useCNNlayers):
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
			else:
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
	else:
		if(activationFunctionType!="none"):
			x = activationFunction(x)
	return x

def getUseLinearSublayers(self, layerIndex):
	result = False
	if(useLinearSublayers):
		if(outputLayerInList):
			if(layerIndex != self.config.numberOfLayers-1):	#final layer does not useLinearSublayers
				result = True
		else:
			result = True
	return result

def weightsSetLayer(self, layerIndex, linear, sign=True):
	if(useSignedWeights):
		weightsSetSignLayer(self, layerIndex, linear, sign)
	if(useCustomWeightInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.normal_(linear.segregatedLinear.weight, mean=Wmean, std=WstdDev)
		else:
			nn.init.normal_(linear.weight, mean=Wmean, std=WstdDev)
	if(useCustomBiasInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.constant_(linear.segregatedLinear.bias, 0)
		else:
			nn.init.constant_(linear.bias, 0)

def weightsFixLayer(self, layerIndex, linear, sign=True):
	#if(not trainLastLayerOnly):
	if(not usePositiveWeightsClampModel):
		weightsSetSignLayer(self, layerIndex, linear, sign)
			
def weightsSetSignLayer(self, layerIndex, linear, sign=True):
	if(getUseLinearSublayers(self, layerIndex)):
		weights = linear.segregatedLinear.weight #only sign weights allowed
		weights = pt.abs(weights)
		if(not sign):
			weights = -weights
		linear.segregatedLinear.weight = pt.nn.Parameter(weights)
	else:
		weights = linear.weight #only sign weights allowed
		weights = pt.abs(weights)
		if(not sign):
			weights = -weights
		linear.weight = pt.nn.Parameter(weights)
	if(debugUsePositiveWeightsVerify):
		if(getUseLinearSublayers(self, layerIndex)):
			weights = linear.segregatedLinear.weight
			print("weights = ", weights)
		else:
			weights = linear.weight
			print("weights = ", weights)		
		
def weightsSetPositiveModel(self):
	if(useSignedWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)

class OffsetReLU(nn.Module):
	def __init__(self, offset):
		super(OffsetReLU, self).__init__()
		self.offset = offset

	def forward(self, x):
		if(debugPrintActivationOutput):
			print("OffsetReLU: x = ", x)
		#print("self.offset = ", self.offset)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x

class OffsetSoftmax(nn.Module):
	def __init__(self, offset):
		super(OffsetSoftmax, self).__init__()
		self.offset = offset
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		if(debugPrintActivationOutput):
			print("OffsetSoftmax: x = ", x)
		#print("self.offset = ", self.offset)
		x = self.softmax(x)
		if(debugPrintActivationOutput):
			print("OffsetSoftmax: x after softmax = ", x)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x





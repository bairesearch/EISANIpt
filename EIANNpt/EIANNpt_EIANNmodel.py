"""EIANNpt_EIANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EIANNpt excitatory inhibitory artificial neural network model

implementation note:
- a relU function is applied to both E and I neurons (so their output will always be positive)
- the weight matrices with inhibitory neuron input (ie Ei and Ii) are always set to negative

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers
if(trainingUpdateImplementation == "backprop"):
	import EIANNpt_VICRegANNloss

class EIANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples
		
class EIANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		configE = config
		configI = config
		configE.hiddenLayerSize = hiddenLayerSizeE
		configI.hiddenLayerSize = hiddenLayerSizeI
		
		layersLinearListEe = []
		layersLinearListEi = []
		layersLinearListIe = []
		layersLinearListIi = []
		layersActivationListE = []
		layersActivationListI = []
		for layerIndex in range(config.numberOfLayers):	
			featuresFanIn = False
			if(inhibitoryNeuronOutputPositive):
				if(layerIndex > 0):	#and layerIndex < config.numberOfLayers-1
					featuresFanIn = True
			inFeaturesMatchHidden = False
			inFeaturesMatchOutput = False
			if(inhibitoryNeuronInitialisationMethod=="intermediaryInterneuron"):
				if(layerIndex == 0):	
					inFeaturesMatchHidden = True	#inhibitoryInterneuronFirstLayer	#CHECKTHIS: set first inhibitory layer size to input layer size (ensure zEi will be a same shape as zEe)
				if(layerIndex == config.numberOfLayers-1):
					inFeaturesMatchOutput = True	#inhibitoryInterneuronLastLayer	
			linearEe = ANNpt_linearSublayers.generateLinearLayerMatch(self, layerIndex, config, sign=True, featuresFanIn=featuresFanIn)	#excitatory neuron excitatory input
			linearEi = ANNpt_linearSublayers.generateLinearLayerMatch(self, layerIndex, config, sign=False, featuresFanIn=featuresFanIn, inFeaturesMatchHidden=inFeaturesMatchHidden, inFeaturesMatchOutput=inFeaturesMatchOutput)	#excitatory neuron inhibitory input
			linearIe = ANNpt_linearSublayers.generateLinearLayerMatch(self, layerIndex, config, sign=True, featuresFanIn=featuresFanIn)	#inhibitory neuron excitatory input
			linearIi = ANNpt_linearSublayers.generateLinearLayerMatch(self, layerIndex, config, sign=False, featuresFanIn=featuresFanIn)	#inhibitory neuron inhibitory input
			layersLinearListEe.append(linearEe)
			layersLinearListEi.append(linearEi)
			layersLinearListIe.append(linearIe)
			layersLinearListIi.append(linearIi)
			
			activationE = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config, positive=True)
			activationI = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config, positive=(not invertActivationFunctionForEIneurons))
			layersActivationListE.append(activationE)
			layersActivationListI.append(activationI)
		self.layersLinearEe = nn.ModuleList(layersLinearListEe)
		self.layersLinearEi = nn.ModuleList(layersLinearListEi)
		self.layersLinearIe = nn.ModuleList(layersLinearListIe)
		self.layersLinearIi = nn.ModuleList(layersLinearListIi)
		self.layersActivationE = nn.ModuleList(layersActivationListE)
		self.layersActivationI = nn.ModuleList(layersActivationListI)
	
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunction = nn.CrossEntropyLoss()
		else:
			self.lossFunction = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)
				
	def forward(self, trainOrTest, x, y, optim=None, l=None):
		if(useLinearSublayers):
			x = x.unsqueeze(dim=1)
		xE = x
		if(inhibitoryNeuronInitialisationMethod=="useInputActivations"):
			if(inhibitoryNeuronOutputPositive):
				xI = pt.zeros_like(x)	#there is no inhibitory input to first hidden layer in network
			else:
				if(inhibitoryNeuronSwitchActivation):
					#assumes datasetNormaliseStdAvg;
					xE = pt.where(x > 0, x, pt.tensor(0))
					xI = pt.where(x < 0, -x, pt.tensor(0))
					#print("xE = ", xE)
					#print("xI = ", xI)
				else:
					xI = x
		else:
			xI = pt.zeros_like(x)	#there is no inhibitory input to first hidden layer in network
		for layerIndex in range(self.config.numberOfLayers):
			if(debugSanityChecks):
				print("\nlayerIndex = ", layerIndex)
			if(EIANNlocalLearning):
				xE = xE.detach()
				xI = xI.detach()
	
			xPrevE = xE
			xPrevI = xI
			zIe = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xPrevE, self.layersLinearIe[layerIndex], sign=True)	#inhibitory neuron excitatory input
			zIi = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xPrevI, self.layersLinearIi[layerIndex], sign=False)	#inhibitory neuron inhibitory input
			if(inhibitoryNeuronOutputPositive):
				zI = pt.cat((zIe, zIi), dim=1)
			else:
				zI = zIe + zIi	#sum the positive/negative inputs of the inhibitory neurons
			if(inhibitoryNeuronSwitchActivation):
				xI = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, -zI, self.layersActivationI[layerIndex])	#relU
			else:
				xI = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, zI, self.layersActivationI[layerIndex])	#relU
			if(inhibitoryNeuronInitialisationMethod=="intermediaryInterneuron"):
				zEe = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xPrevE, self.layersLinearEe[layerIndex], sign=True)	#excitatory neuron excitatory input
				zEi = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xI, self.layersLinearEi[layerIndex], sign=False)	#excitatory neuron inhibitory input
			else:
				zEe = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xPrevE, self.layersLinearEe[layerIndex], sign=True)	#excitatory neuron excitatory input
				zEi = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, xPrevI, self.layersLinearEi[layerIndex], sign=False)	#excitatory neuron inhibitory input
			if(inhibitoryNeuronOutputPositive):
				zE = pt.cat((zEe, zEi), dim=1)
			else:
				zE = zEe + zEi	#sum the positive/negative inputs of the excitatory neurons
			xE = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, zE, self.layersActivationE[layerIndex])	#relU
			
			if(inhibitoryNeuronInitialisationMethod=="firstHiddenLayerExcitatoryInputOnly"):
				if(layerIndex==0):
					#normalise via top k (normalise activation sparsity) because there is no inhibitory input
					k = hiddenLayerSizeE//2
					xE = self.deactivateNonTopKneurons(xE, k)
						
			if(debugSmallNetwork):
				print("layerIndex = ", layerIndex)
				print("xE after linear = ", xE)
				print("xI after linear = ", xI)
			if(layerIndex == self.config.numberOfLayers-1):
				if(useInbuiltCrossEntropyLossFunction):
					if(inhibitoryNeuronOutputPositive):
						x = zEe
					else:
						x = zE
				else:
					#NOT SUPPORTED with activationFunction=="positiveMid"
					#xI = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, zI, self.layersActivationI[layerIndex], parallelStreams=False)	#there is no final/output inhibitory layer
					xE = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, zE, self.layersActivationE[layerIndex], parallelStreams=False)
					x = torch.log(x)
			else:
				if(simulatedDendriticBranches):
					x, xIndex = self.performTopK(x)
					
			if(trainOrTest and EIANNlocalLearning):
				if(layerIndex < self.config.numberOfLayers-1):
					if(trainingUpdateImplementation == "backprop"):
						#self.trainLayerBackprop(layerIndex, xE, xI, optim)
						if(debugEIinputs):
							print("zEe = ", zEe)
							print("zEi = ", zEi)
							print("zIe = ", zIe)
							print("zIi = ", zIi)
						self.trainLayerBackprop(layerIndex, zEe, zEi, optim[0])
						self.trainLayerBackprop(layerIndex, zIe, zIi, optim[1])
					elif(trainingUpdateImplementation == "hebbian"):
						self.trainLayerHebbian(layerIndex, xE, xI, zEe, zEi, zIe, zIi, xPrevE, xPrevI)
					
			if(debugSmallNetwork):
				print("x after activation = ", x)
				
		if(useLinearSublayers):
			x = x.squeeze(dim=1)

		loss = self.lossFunction(x, y)
		if(trainLocal):
			self.trainLayerLast(optim[0], loss)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		
		return loss, accuracy

	def calculateActive(self, x):
		xActive = (x>0).float()
		if(EIANNassociationMatrixBatched):
			xActive = xActive.unsqueeze(2)
		else:
			xActive = xActive.unsqueeze(1)
		return xActive
		
	def deactivateNonTopKneurons(self, activations, k):
		#print("activations.shape = ", activations.shape)
		topk_values, topk_indices = pt.topk(activations, k=k, dim=1)
		mask = pt.zeros_like(activations)
		mask.scatter_(1, topk_indices, 1)
		masked_activations = activations * mask
		return masked_activations
	
	def performTopK(self, x):
		xMax = pt.max(x, dim=1, keepdim=False)
		x = xMax.values
		xIndex = xMax.indices
		return x, xIndex

	if(trainingUpdateImplementation == "backprop"):
		
		#derived from VICRegANNpt_VICRegANNloss;
		
		def trainLayerLast(self, optim, loss):
			layerIndex = self.config.numberOfLayers-1
			opt = optim[layerIndex]
			opt.zero_grad()			
			loss.backward()
			opt.step()

		def trainLayerBackprop(self, layerIndex, xE, xI, optim):

			loss = None
			accuracy = 0.0
			
			opt = optim[layerIndex]
			opt.zero_grad()

			loss = EIANNpt_VICRegANNloss.calculatePropagationLossVICRegANN(xE, xI)
			
			loss.backward()
			opt.step()

	elif(trainingUpdateImplementation == "hebbian"):	

		def trainLayerHebbian(self, layerIndex, xE, xI, zEe, zEi, zIe, zIi, xPrevE, xPrevI):
			if(hebbianWeightsUsingEIseparableInputsCorrespondenceMatrix):
				associationMatrixEe = self.calculateAssociationMatrix(layerIndex, zEe, xPrevE)	#all +ve	#shape: [batchSize,] o, i
				if(inhibitoryNeuronInitialisationMethod=="intermediaryInterneuron"):
					associationMatrixEi = self.calculateAssociationMatrix(layerIndex, zEi, xI)	#all -ve	#shape: [batchSize,] o, i
				else:
					associationMatrixEi = self.calculateAssociationMatrix(layerIndex, zEi, xPrevI)	#all -ve	#shape: [batchSize,] o, i
				associationMatrixIe = self.calculateAssociationMatrix(layerIndex, zIe, xPrevE)	#all +ve	#shape: [batchSize,] o, i
				associationMatrixIi = self.calculateAssociationMatrix(layerIndex, zIi, xPrevI)	#all -ve	#shape: [batchSize,] o, i
			else:
				associationMatrixE = self.calculateAssociationMatrix(layerIndex, zE, xPrevE)	#shape: [batchSize,] o, i
				associationMatrixI = self.calculateAssociationMatrix(layerIndex, zI, xPrevI)	#shape: [batchSize,] o, i
				associationMatrixEe = associationMatrixE
				associationMatrixEi = associationMatrixE
				associationMatrixIe = associationMatrixI
				associationMatrixIi = associationMatrixI
			if(debugSanityChecks):
				print("xPrevI = ", xPrevI)
				print("xPrevE = ", xPrevE)
				print("zIe = ", zIe)
				print("zIi = ", zIi)
				print("zEe = ", zEe)
				print("zEi = ", zEi)
				print("zI = ", zI)
				print("zE = ", zE)
				print("xI = ", xI)
				print("xE = ", xE)
				print("associationMatrixIe = ", associationMatrixIe)
				print("associationMatrixIi = ", associationMatrixIi)
				print("associationMatrixEe = ", associationMatrixEe)
				print("associationMatrixEi = ", associationMatrixEi)

			if(not trainInactiveNeurons):
				if(inhibitoryNeuronOutputPositive):
					xEe = xE[:, 0:hiddenLayerSize]
					xEi = xE[:, hiddenLayerSize:]
					xIe = xI[:, 0:hiddenLayerSize]
					xIi = xI[:, hiddenLayerSize:]
					#print("associationMatrixEe.shape = ", associationMatrixEe.shape)
					#print("xEe.shape = ", xEe.shape)
					associationMatrixEe = associationMatrixEe*self.calculateActive(xEe)
					associationMatrixEi = associationMatrixEi*self.calculateActive(xEi)
					associationMatrixIe = associationMatrixIe*self.calculateActive(xIe)
					associationMatrixIi = associationMatrixIi*self.calculateActive(xIi)
				else:
					associationMatrixEe = associationMatrixEe*self.calculateActive(xE)
					associationMatrixEi = associationMatrixEi*self.calculateActive(xE)
					associationMatrixIe = associationMatrixIe*self.calculateActive(xI)
					associationMatrixIi = associationMatrixIi*self.calculateActive(xI)
			if(EIANNlocalLearningApplyError):
				errorE = self.calculateError(zEe, zEi, True)	#ie zE	#shape: batchSize, E
				errorI = self.calculateError(zIe, zIi, False)	#ie zI	#shape: batchSize, I
				hebbianMatrixEe = self.calculateHebbianMatrix(associationMatrixEe, errorE, False)	#if +ve error, want to decrease Ee (ie +ve) weights
				hebbianMatrixEi = self.calculateHebbianMatrix(associationMatrixEi, errorE, True)	#if +ve error, want to increase Ei (ie -ve) weights
				hebbianMatrixIe = self.calculateHebbianMatrix(associationMatrixIe, errorI, False)	#if +ve error, want to decrease Ie (ie +ve) weights
				hebbianMatrixIi = self.calculateHebbianMatrix(associationMatrixIi, errorI, True)	#if +ve error, want to increase Ii (ie -ve) weights
			else:
				hebbianMatrixEe = associationMatrixEe
				hebbianMatrixEi = associationMatrixEi
				hebbianMatrixIe = associationMatrixIe
				hebbianMatrixIi = associationMatrixIi

			if(inhibitoryNeuronInitialisationMethod!="firstHiddenLayerExcitatoryInputOnly" or layerIndex > 0):
				self.trainWeightsLayer(layerIndex, hebbianMatrixEe, self.layersLinearEe[layerIndex])
				self.trainWeightsLayer(layerIndex, hebbianMatrixEi, self.layersLinearEi[layerIndex])
				self.trainWeightsLayer(layerIndex, hebbianMatrixIe, self.layersLinearIe[layerIndex])
				if(inhibitoryNeuronInitialisationMethod!="intermediaryInterneuron" or layerIndex > 0):
					self.trainWeightsLayer(layerIndex, hebbianMatrixIi, self.layersLinearIi[layerIndex])

		def calculateAssociationMatrix(self, layerIndex, z, xPrev):
			if(useLinearSublayers):
				z = pt.squeeze(z, dim=1)		#assume linearSublayersNumber=1
				xPrev = pt.squeeze(xPrev, dim=1)	#assume linearSublayersNumber=1

			if(associationMatrixMethod=="useInputsAndOutputs"):
				if(EIANNassociationMatrixBatched):
					#retain batchSize dim in associationMatrix
					z = z.unsqueeze(2)  # Shape will be (batchSize, z, 1)
					xPrev = xPrev.unsqueeze(1)  # Shape will be (batchSize, 1, xPrev)
					associationMatrix = z * xPrev  # Resultant shape will be (batchSize, z, xPrev)
				else:
					associationMatrix = pt.matmul(pt.transpose(z, 0, 1), xPrev)
				#print("associationMatrix.shape = ", associationMatrix.shape)
			elif(associationMatrixMethod=="useInputsAndWeights"):
				weights = self.layersLinearIi[layerIndex].weight
				#print("weights.shape = ", weights.shape)
				#print("xPrev.shape = ", xPrev.shape)
				if(EIANNassociationMatrixBatched):
					weights = weights.unsqueeze(0)	#Shape will be (1, z, xPrev)
					xPrev = xPrev.unsqueeze(1)	 #Shape will be (batchSize, 1, xPrev)
				else:
					printe("calculateAssociationMatrix error: associationMatrixMethod=useInputsAndWeights currently requires EIANNassociationMatrixBatched")
				associationMatrix = xPrev*weights	# Resultant shape will be (batchSize, z, xPrev)
				#print("associationMatrix.shape = ", associationMatrix.shape)

			return associationMatrix

		def calculateErrorSign(self, e, i):
			error = calculateError(e, i)
			errorSign = pt.sign(error).float()
			return errorSign

		def calculateError(self, e, i, isExcitatoryNeuron):		
			if(trainThreshold=="positive"):
				if(inhibitoryNeuronSwitchActivation and not isExcitatoryNeuron):
					error = e + i + trainThresholdPositiveValue
				else:
					error = e + i - trainThresholdPositiveValue
			elif(trainThreshold=="zero"):
				error = e + i	#e is +ve, i is -ve
			return error

		def calculateHebbianMatrix(self, associationMatrix, error, sign):
			associationMatrix = pt.abs(associationMatrix)
			#print("associationMatrix.shape = ", associationMatrix.shape)
			#print("error.shape = ", error.shape)
			if(EIANNassociationMatrixBatched):
				error = pt.unsqueeze(error, dim=2)
				if(inhibitoryNeuronOutputPositive):
					hebbianMatrix = associationMatrix*error*-1	
				else:
					if(sign):
						hebbianMatrix = associationMatrix*error*1
					else:
						hebbianMatrix = associationMatrix*error*-1	
				hebbianMatrix = pt.mean(hebbianMatrix, dim=0) #average error over batchSize
			else:
				error = pt.mean(error, dim=0) #average error over batchSize	#associationMatrix has already been averaged over batchSize
				error = pt.unsqueeze(error, dim=1)
				if(inhibitoryNeuronOutputPositive):
					hebbianMatrix = associationMatrix*error*-1
				else:
					if(sign):
						hebbianMatrix = associationMatrix*error*1
					else:
						hebbianMatrix = associationMatrix*error*-1
			return hebbianMatrix

		def trainWeightsLayer(self, layerIndex, hebbianMatrix, layerLinear):
			#use local hebbian learning rule - CHECKTHIS
			layerWeights = layerLinear.weight

			weightUpdate = hebbianMatrix*EIANNlocalLearningRate
			layerWeights = layerWeights + weightUpdate

			layerLinear.weight = pt.nn.Parameter(layerWeights)


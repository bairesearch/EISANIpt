"""EIANNpt_VICRegANNloss.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EIANNpt Variance-Invariance-Covariance Regularization artificial neural network (VICRegANN) loss

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *

def calculatePropagationLossVICRegANN(A1, A2):

	if(partiallyAlignLayer):
		layerSize = A1.shape[1]
		maskSize = (layerSize - layerSize*partiallyAlignLayerFraction)
		mask = pt.range(0, maskSize, dtype=pt.int64).to(device)
		A1 = pt.index_fill(A1, dim=1, index=mask, value=partiallyAlignLayerIgnoreValue)
		A2 = pt.index_fill(A2, dim=1, index=mask, value=partiallyAlignLayerIgnoreValue)
		#print("A1 = ", A1)
		
	if(trainMostAlignedNeurons):
		#A1bool = pt.gt(A1, 0)
		#A2bool = pt.gt(A2, 0)
		#neuronsAligned = pt.eq(A1, A2)
		AdotProduct = pt.multiply(A1, A2)
		
		if(trainMostAlignedNeuronsMethod == "softmax"):
			AdotProductThresholded = pt.nn.functional.softmax(AdotProduct, dim=1)
		elif(trainMostAlignedNeuronsMethod == "thresholded"):
			AdotProductThresholded = pt.gt(AdotProduct, trainMostAlignedNeuronsThresholdMin).float()
		elif(trainMostAlignedNeuronsMethod == "topk"):
			topkRes = pt.topk(AdotProduct, trainMostAlignedNeuronsTopK)
			AdotProductMask = pt.zeros(AdotProduct.shape).to(device)
			AdotProductMask.scatter_(-1, topkRes.indices, topkRes.values)
			AdotProductThresholded = AdotProductMask
			#print("AdotProductThresholded = ", AdotProductThresholded)
	
		A1 = pt.multiply(A1, AdotProductThresholded)
		A2 = pt.multiply(A2, AdotProductThresholded)
		
	#invariance loss
	matchedClassPairSimilarityLoss = calculateSimilarityLoss(A1, A2)

	if(vicregSimilarityLossOnly):
		loss = lambdaHyperparameter*matchedClassPairSimilarityLoss
	else:
		#variance loss
		batchVariance1 = calculateVarianceBatch(A1)
		batchVariance2 = calculateVarianceBatch(A2)
		varianceLoss = pt.mean(pt.nn.functional.relu(1.0 - batchVariance1)) + pt.mean(pt.nn.functional.relu(1.0 - batchVariance2))

		#covariance loss
		covariance1matrix = calculateCovarianceMatrix(A1)
		covariance2matrix = calculateCovarianceMatrix(A2)
		covarianceLoss = calculateCovarianceLoss(covariance1matrix) + calculateCovarianceLoss(covariance2matrix)

		loss = lambdaHyperparameter*matchedClassPairSimilarityLoss + muHyperparameter*varianceLoss + nuHyperparameter*covarianceLoss

	if(debugVICRegLoss):
		print("A1 = ", A1)
		print("A2 = ", A2)	
		print("batchVariance1 = ", batchVariance1)
		print("varianceLoss = ", varianceLoss)	
		print("matchedClassPairSimilarityLoss = ", matchedClassPairSimilarityLoss)
		print("covariance1matrix = ", covariance1matrix)
		print("covarianceLoss = ", covarianceLoss)
		print("loss = ", loss)
		exit()
		
	return loss
	
def calculateVarianceBatch(A):
	batchVariance = pt.sqrt(reduceVariance(A, dim=0) + 1e-04)
	#print("batchVariance = ", batchVariance)
	return batchVariance
	
def calculateSimilarityLoss(A1, A2):
	#Apair = pt.stack([A1, A2])
	#matchedClassPairVariance = reduceVariance(Apair, axis=0)
	similarityLoss = calculateLossMeanSquaredError(A1, A2)
	return similarityLoss
	
def calculateCovarianceMatrix(A):
	#covariance = calculateCovarianceMean(A)
	A = A - pt.mean(A, dim=0)
	batchSize = A.shape[0]
	covarianceMatrix = (pt.matmul(pt.transpose(A, 0, 1), A)) / (batchSize - 1.0)
	return covarianceMatrix

def calculateCovarianceLoss(covarianceMatrix):
	numberOfDimensions = covarianceMatrix.shape[0]	#A1.shape[1]
	covarianceLoss = pt.sum(pt.pow(zeroOnDiagonalMatrixCells(covarianceMatrix), 2.0))/numberOfDimensions
	return covarianceLoss

def zeroOnDiagonalMatrixCells(covarianceMatrix):
	numberVariables = covarianceMatrix.shape[0]
	diagonalMask = pt.eye(numberVariables).to(device)
	diagonalMaskBool = pt_cast(diagonalMask, pt.bool)
	diagonalMaskBool = pt.logical_not(diagonalMaskBool)
	diagonalMask = pt_cast(diagonalMaskBool, pt.float)
	covarianceMatrix = pt.multiply(covarianceMatrix, diagonalMask)
	return covarianceMatrix

def reduceVariance(A, dim=0):
	varCustom = True
	if(varCustom):
		return varianceUnbiasedCustom(A, dim, keepdim=False)
	else:
		return pt.var(A, dim, keepdim=False, unbiased=True)	#CHECKTHIS: unbiased

def varianceUnbiasedCustom(input, dim, keepdim=False):
	#https://discuss.pytorch.org/t/inconsistency-between-pytorch-and-tensorflows-variance-functions-results-and-how-pytorch-implements-it-using-the-summation-function/123176
	input_means = pt.mean(input, dim=dim, keepdim=True)
	squared_deviations = squared_difference(input, input_means)
	variance = pt.mean(squared_deviations, dim=dim, keepdim=keepdim)	
	return variance

def calculateLossMeanSquaredError(y_pred, y_true):
	loss = pt.mean(squared_difference(y_pred, y_true))
	return loss
	
def squared_difference(input, target):
	#print("input.shape = ", input.shape)
	#print("target.shape = ", target.shape)
	return (input - target) ** 2

def pt_cast(t, type):
	t = t.type(type)
	return t

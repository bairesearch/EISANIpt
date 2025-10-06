"""EISANIpt_EISANI_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
EISANIpt globalDefs

"""

import math

#debug parameters;
debugEISANIoutputs = False
printEISANImodelProperties = True

#testharness parameters;
useDefaultNumNeuronSegmentsParam = True	#default: True (use low network width)
useDefaultSegmentSizeParam = True	#default: True (use moderate segment size/num synapses)
useDefaultNumLayersParam = True	#default: True (use low num layers)
useInitOrigParam = False	#use original test parameters

#dataset selection;
useTabularDataset = True
useImageDataset = False
useNLPDataset = False	#aka useSequenceDataset

#architecture selection;
if(useNLPDataset):
	useSequentialSANI = True	#mandatory: True	#sequentially activated neuronal input (else use summation activated neuronal input)
	useStochasticUpdates = False	#mandatory: False (not currently supported by useSequentialSANI)
else:
	useSequentialSANI = False	#mandatory: False
	useStochasticUpdates = False	#default: False (when True, learning uses stochastic trials)	#stochastic update option: try random independent connection changes
	#useStochasticUpdates is the new default method for useDynamicGeneratedHiddenConnections=False (uses output layer backprop learning only)

#init derived params (do not modify here);
useSequentialSANIactivationStrength = False
limitOutputConnectionsBasedOnAccuracySoftmax = False
evalStillTrainOutputConnections = False
debugOnlyPrintStreamedWikiArticleTitles = False
useSlidingWindow = False
targetActivationSparsityFraction = -1
numberNeuronSegmentsGeneratedPerSample = -1
EISANICNNrandomlySelectInputBinaryStates = False
useDynamicGeneratedCNNConnections = False
EISANICNNarchitectureDensePretrained = False

if(useTabularDataset):
	useContinuousVarEncodeMethod = "grayCode"	#use graycode to encode continuous vars into binary (else use thermometer encoding)
elif(useImageDataset):
	debugEISANICNNdynamicallyGenerateLinearInputFeatures = False	#print nextLinearCol - number of linear layer input encoding features used
	debugEISANICNNprintKernels = True
	
	EISANICNNarchitectureDivergeAllKernelPermutations = False	#orig: True
	EISANICNNarchitectureDivergeLimitedKernelPermutations = False	#default 1: True
	EISANICNNarchitectureSparseRandom = True	#default 2: True
	EISANICNNarchitectureDenseRandom = False
	EISANICNNarchitectureDensePretrained = False
	
	CNNkernelSize = 3
	CNNstride = 1
	CNNmaxPool = True
	encodedFeatureSizeDefault = 12800000*math.ceil(1/2)	#input linear layer encoded features are dynamically generated from historic active neurons in final CNN layer	#configured for numberOfConvlayers=2
	trainNumberOfEpochsHigh = False	#default: False
	EISANICNNdynamicallyGenerateFFInputFeatures = False
	if(EISANICNNarchitectureSparseRandom):
		EISANICNNuseBinaryInput = True
		EISANICNNrandomlySelectInputBinaryStates = False	#default: False
		if(EISANICNNrandomlySelectInputBinaryStates):
			EISANICNNinputChannelThreshold = -1	#no fixed threshold
			EISANICNNnumberOfRandomlySelectedInputBinaryStates = 16	#default: 256	#binary input values are generated based on the probability distribution given by the input float values
		else:
			EISANICNNinputChannelThreshold = 0.5 #default: 0.5
			EISANICNNnumberOfRandomlySelectedInputBinaryStates = 1
		EISANICNNcontinuousVarEncodingNumBits = 1	#mandatory: 1		#number of bits to encode image pixels
		EISANITABcontinuousVarEncodingNumBitsAfterCNN = 1	#mandatory: 1
		EISANICNNactivationFunction = True	#mandatory
		EISANICNNpaddingPolicy = 'same'
		EISANICNNnumberKernels = 64	#default: 16
		EISANICNNkernelSizeSANI = 64	#default: 32
		numberOfConvlayers = 4	#rest will be FF	#default: 2, 4, 6
		EISANICNNmaxPoolEveryQLayers = 2	#orig: 1	#default: 1
		useDynamicGeneratedCNNConnections = True
		if(useDynamicGeneratedCNNConnections):
			useDynamicGeneratedCNNConnectionsVectorised = True	#execute entire batch simultaneously
			useDynamicGeneratedCNNConnectionsUniquenessChecks = True
	elif(EISANICNNarchitectureDenseRandom):
		EISANICNNuseBinaryInput = False
		EISANICNNcontinuousVarEncodingNumBits = 1	#mandatory: 1		#number of bits to encode image pixels
		useContinuousVarEncodeMethodAfterCNN = "grayCode"
		EISANITABcontinuousVarEncodingNumBitsAfterCNN = 8	#default: 8
		EISANICNNoutputChannelThreshold = 0.5 #default: 0.5	#if EISANITABcontinuousVarEncodingNumBitsAfterCNN=1
		EISANICNNactivationFunction = True	#mandatory
		EISANICNNpaddingPolicy = 'same'
		numberOfConvlayers = 6	#rest will be FF	#default: 2, 4, 6
		EISANICNNmaxPoolEveryQLayers = 2	#orig: 1	#default: 1
	elif(EISANICNNarchitectureDensePretrained):
		EISANICNNuseBinaryInput = False
		EISANICNNcontinuousVarEncodingNumBits = 1	#mandatory: 1		#number of bits to encode image pixels
		useContinuousVarEncodeMethodAfterCNN = "grayCode"
		EISANITABcontinuousVarEncodingNumBitsAfterCNN = 1	#default: 8
		EISANICNNoutputChannelThreshold = 0.5 #default: 0.5	#if EISANITABcontinuousVarEncodingNumBitsAfterCNN=1
		EISANICNNactivationFunction = True	#mandatory
		EISANICNNpaddingPolicy = 'same'
		numberOfConvlayers = 6	#rest will be FF	#default: 2, 4, 6
		EISANICNNmaxPoolEveryQLayers = 2	#orig: 1	#default: 1
	elif(EISANICNNarchitectureDivergeAllKernelPermutations):
		EISANICNNuseBinaryInput = True
		EISANICNNinputChannelThreshold = 0.5 #default: 0.5	#if EISANITABcontinuousVarEncodingNumBitsAfterCNN=1
		useContinuousVarEncodeMethod = "grayCode"
		EISANICNNcontinuousVarEncodingNumBits = 1	#mandatory: 1		#number of bits to encode image pixels
		EISANITABcontinuousVarEncodingNumBitsAfterCNN = 1	#mandatory: 1
		EISANICNNdynamicallyGenerateFFInputFeatures = True	#default: True	#input FF layer encoded features are dynamically generated from historic active neurons in final CNN layer	#EISANICNNdynamicallyGenerateFFInputFeatures requires EISANICNNoptimisationSparseConv and numberOfConvlayers > 1
		EISANICNNnumberKernelOrientations = -1		
		EISANICNNkernelEdgesTernary = False	#Binary = +1, -1 weights	#too many permutations with ternary weights 
		EISANICNNactivationFunction = True	#mandatory
		EISANICNNpaddingPolicy = 'none'
		EISANICNNoptimisationSparseConv = True	#default: True	#only apply convolution to channels with at least 1 on bit
		EISANICNNoptimisationAssumeInt8 = False	#default: False	#if True; cnn operations (conv2d/maxpool2d) are not currently implemented on CuDNN, so will still be temporarily converted to float
		numberOfConvlayers = 2	#rest will be FF	#default: 2, 4, 6
		EISANICNNmaxPoolEveryQLayers = 1	#orig: 1	#default: 1
	elif(EISANICNNarchitectureDivergeLimitedKernelPermutations):
		EISANICNNuseBinaryInput = False
		EISANICNNcontinuousVarEncodingNumBits = 1	#mandatory: 1	#retain float do not convert to bits
		useContinuousVarEncodeMethodAfterCNN = "grayCode"
		EISANITABcontinuousVarEncodingNumBitsAfterCNN = 1	#default: 1	#orig: 1
		EISANICNNnumberKernelOrientations = 8	#default 8 (45 degree increments)
		EISANICNNoutputChannelThreshold = 0.5 #default: 0.5
		EISANICNNactivationFunction = True	#default: True
		EISANICNNpaddingPolicy = 'same'	#default: same, orig: none
		EISANICNNpaddingMode = 'reflect'	#default: reflect #or zeros
		useOrigCNNParam = True
		if(useOrigCNNParam):
			EISANICNNkernelEdgesSharp = False
			EISANICNNkernelEdgesTernary = True	#Ternary = +1, 0, -1 weights 	#no effect on number of permutations	#mandatory: currently required for !EISANICNNkernelEdgesSharp
			EISANICNNkernelEdges = True
			EISANICNNkernelCorners = False
			EISANICNNkernelCentroids = False
		else:
			EISANICNNkernelEdgesSharp = True
			EISANICNNkernelEdgesTernary = True	#default: True	#Ternary = +1, 0, -1 weights 	#no effect on number of permutations
			EISANICNNkernelEdges = True
			EISANICNNkernelCorners = True
			EISANICNNkernelCentroids = True
		EISANICNNsigma = 0.85	# Gaussian envelope sigma
		EISANICNNtau = 0.35	# tanh slope (smaller = sharper)
		EISANICNNkernelZeroBandHalfWidth = 1e-6	 # ternary soft band
		EISANICNNcentroidSigma = max(0.5, EISANICNNsigma * 0.75)
		EISANICNNcentroidScaleRatio = 1.6
		numberOfConvlayers = 4	#rest will be FF	#default: 2, 4, 6
		if(numberOfConvlayers == 2):
			EISANICNNmaxPoolEveryQLayers = 1	#orig: 1	#default: 1
		else:
			EISANICNNmaxPoolEveryQLayers = 2	#default: 2
		
	numberOfFFLayers = 3
		
elif(useNLPDataset):
	debugOnlyPrintStreamedWikiArticleTitles = False

	useSlidingWindow = True	#emulate SANI (sequentially activated neuronal input) requirement by reusing neuron activations from previous sliding window iteration	#mandatory	#useNeuronActivationMemory
	#enforceSequenceContiguity = True	#FUTURE: perform sequence contiguity test for generated synaptic inputs (see SANI specification)
	useDefaultSegmentSizeParam = False	#currently use smaller number of requisite active connections
	if(useSequentialSANI):
		useNLPcharacterInput = False	#default: False - use token input
	else:
		useNLPcharacterInput = True		#default: True - use character input
	useNLPcharacterInputBasic = True	#if True: only use a basic lowercase+punctuation character set of 30 chars, else if False: use a full printable subset of ASCII-128
	if(useNLPcharacterInput):
		useContinuousVarEncodeMethod = "onehot"	#just convert character id directly to onehot vector
		if(useNLPcharacterInputBasic):
			NLPcharacterInputBasicSet = [' ', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.','(',')', ',']	#select 31 characters for normalcy
			NLPcharacterInputSetLen = len(NLPcharacterInputBasicSet)+1	#32	# 0 reserved for PAD (NLPcharacterInputPadTokenID)
		else:
			NLPcharacterInputSetLen = 98	  # full printable subset of ASCII-128	# 0 reserved for PAD (NLPcharacterInputPadTokenID)
		EISANINLPcontinuousVarEncodingNumBits = NLPcharacterInputSetLen
		
		contextSizeMax = 128*4	#default: 512	#production: 512*4	#assume approx 4 characters per BERT token
		numberOfClasses = NLPcharacterInputSetLen
	else:	
		bertModelName = "bert-base-uncased"	#bertModelName = "bert-large-uncased"
		bertNumberTokenTypes = 30522	#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")	print(len(tokenizer))
		if(useSequentialSANI):
			useTokenEmbedding = False
			useContinuousVarEncodeMethod = "onehot"	#use thermometer encoding (as already encoded)
			EISANINLPcontinuousVarEncodingNumBits = bertNumberTokenTypes
		else:
			useTokenEmbedding = True
			useContinuousVarEncodeMethod = "thermometer"	#use thermometer encoding (as already encoded)
			EISANINLPcontinuousVarEncodingNumBits = 3	#default: 3; -1.0, 0, +1.0	#alternative eg 5; -1, -0.5, 0, 0.5, 1.0
			embeddingSize = 768	#embeddingSize = 1024
		
		contextSizeMax = 64	#efficient: 64 #default: 128	#depends on sequentialSANItimeInvariance (uses tokens in inference further than training), sequentialSANIoverlappingSegments (linear instead of exponential time invariance across layers), memory efficiency (less tokens means less serial and more parallel processing), and average number of tokens per sentence (to ensure always sampling tokens across entire sentences)
		numberOfClasses = bertNumberTokenTypes
	sequenceLength = contextSizeMax
	NLPcharacterInputPadTokenID = 0	#must be same as bert pad token id	#assert bert_tokenizer.pad_token_id == NLPcharacterInputPadTokenID

if(useSequentialSANI):
	debugSequentialSANIactivationsLoops = False
	debugSequentialSANIactivationsMemory = False
	debugSequentialSANIactivations = False
	debugDynamicallyGenerateLayerNeurons = False
	debugGenerateConnectionsBeforePropagating = False	#will artificially increase prediction accuracy
	debugSequentialSANIpropagationVerify = False
	
	evalOnlyUsingTimeInvariance = False  #default: True	#orig: False	#Assume EISANImodel is created and trained with sequentialSANItimeInvariance and useSequentialSANIactivationStrength disabled, and then loaded with sequentialSANItimeInvariance and useSequentialSANIactivationStrength enabled for inference
	if(evalOnlyUsingTimeInvariance):
		evalStillTrainOutputConnections = False	#default: False	#orig: False	#still train output connections during eval phase - enables tweaking of output predictions based on time invariant network propagation during eval phase
	
	useConnectionWeights = False	#mandatory: False - use 1D index tensors rather than standard weight tensors	#note useSequentialSANI:!useConnectionWeights uses a more memory efficient pruning method (actual hidden/output matrix shapes are modified)
	if(not useConnectionWeights):	
		blockInitCapacity = 1000	#initial number of hidden neurons
		blockExpansionSize = 1000	#expansion additional hidden neurons
	numberOfLayers = 6	#supports relationships/associations across approx 2^6 (numberOfSegmentsPerNeuron^numberOfLayers) tokens with contiguous inputs (no missing/gap tokens)
	numberOfSynapsesPerSegment = 1	#mandatory: 1	#FUTURE; with numberOfSynapsesPerSegment=1; consider updating the connectivity implementation to use simple one-to-one indexing (of previous layer neurons) rather than sparse tensors (modify compute_layer_sequentialSANI and sequentialSANI_dynamic_hidden_growth_pairwise)
	#for redundancy; numberOfSynapsesPerSegment = numberOfLayers	#number of layers in network
	sequentialSANIoverlappingSegments = True	#default: True	#orig: True	#False: contiguous segments only #disable for algorithm debug (trace activations/network gen)	#required for non-even tree structure for neuron input	#recommended for sequentialSANIsegmentRequirement == "both", optional for "any"/"first"
	sequentialSANItimeInvariance = False	 #default: True  #enables redundancy more immediate tokens, closer to timeIndex of the last token in the segment
	if(sequentialSANItimeInvariance):
		debugSequentialSANItimeInvarianceDisable = False	#disable time invariance for temp debug, but still print all time invariance (distance/proximity) calculations in useSequentialSANIactivationStrength
		debugSequentialSANItimeInvarianceVerify = False
		sequentialSANItimeInvarianceFactor = 2.0	#default: 2.0	#minimum: 1.0
		#maxActivationRecallTime = 100	#max number tokens between poximal and distal segment (supports missing/gap tokens)	#heursitic: > numberOfSegmentsPerNeuron^numberOfLayers tokens
		inputLayerTimeInvariance = False	#default: False	#True: time invariance is applied to input layer also (ie non-contiguous input layer tokens)
		useSequentialSANIactivationStrength = True	#default: True	#else use binary activations only	#requires hidden neuron activation function threshold of activation strength tweaked based on sequentialSANIsegmentsPartialActivationCount/sequentialSANIsegmentsPartialActivationDistance
		if(useSequentialSANIactivationStrength):
			debugSequentialSANIactivationsStrength = False
			sequentialSANIsegmentsPartialActivationCount = True	 #default: True #enables redundancy (not every segment needs to be completely represented; some can only contain less activated nodes in their seg0 or seg1, recursively) #requires non-symmetrical tree structure
			sequentialSANIsegmentsPartialActivationDistance = True	 #default: True  #enables redundancy (favour more immediate tokens, closer to timeIndex of the last token in the segment)	#required for sequentialSANItimeInvariance
			sequentialSANIsegmentRequirement = "any"	#both (orig): both segments must be active, any (default): only one segment must be active, first: only last segment must be active, none: no segments must be active (linear)			
			if(sequentialSANIsegmentRequirement == "both"):
				segmentActivationFractionThreshold = 0.75	#default: tune hyperparameter #orig = 0.75	 #proportion of synapses (newly activated lower layer SANI nodes) which must be active for a segment to be active	#CHECKTHIS threshold
			elif(sequentialSANIsegmentRequirement == "any" or sequentialSANIsegmentRequirement == "first"):
				segmentActivationFractionThreshold = 0.5	#must be <= 0.5
			elif(sequentialSANIsegmentRequirement == "none"):
				segmentActivationFractionThreshold = 0.5
			sequentialSANIinhibitoryTopkSelection = False	#default: False	#perform a topk selection of activations based on activations strengths
			if(sequentialSANIinhibitoryTopkSelection):
				debugSequentialSANIinhibitoryTopkSelection = False
				sequentialSANIinhibitoryTopkSelectionKfraction = 0.005	#fraction of neurons on layer to select
	else:
		useSequentialSANIactivationStrength = False	#mandatory: False (sequentialSANIsegmentsPartialActivationDistance requires sequentialSANItimeInvariance)
	numberOfSegmentsPerNeuron = 2
	sequentialSANIsegmentIndexProximal = 0
	sequentialSANIsegmentIndexDistal = 1
	useEIneurons = False	#mandatory: False
	useDynamicGeneratedHiddenConnections = True	#mandatory: True
	if(debugGenerateConnectionsBeforePropagating):
		generateConnectionsAfterPropagating = False 	#debug: False (generate hidden/output connections before propagating; creates hidden neurons/connections and output connections more quickly for debug)
	else:
		generateConnectionsAfterPropagating = True	#default: True
	useSparseHiddenMatrix = True	#default: True	#use sparse tensors to store connections (else use dense tensors)	#mandatory for any reasonably sized EISANI network
	useSparseOutputMatrix = True	#default: True	#required to prevent very large output matrix multiplications (eg hiddenLayerSizeSANI~100000 x bertNumberTokenTypes~30522)
	#hiddenLayerSizeSANImax	#default: #heuristic: number of 5-grams=1.18 billion (see Google ngrams) 	#max = bertNumberTokenTypes^numberOfLayers (not all permutations are valid)		#bertNumberTokenTypes*2
	
	#for print only;
	EISANITABcontinuousVarEncodingNumBits = -1	#not used
	recursiveLayers = False	#not used
	recursiveSuperblocksNumber = 1	#not used
	useCPU = False
else:
	debugStochasticUpdates = False
	debugSequentialSANIactivationsLoops = False
	debugEISANIfractionActivated = False	#print fractionActive of each layer
	debugEISANIdynamicUsage = False	#print neuronSegmentAssignedMask available.numel() - number of FF layer hidden features used

	if(useStochasticUpdates):
		useStochasticUpdatesHiddenUnitLearning = False	#default: False	#orig: True
		stochasticHiddenUpdatesPerBatch = 100	#default: 10	#number of hidden unit stochastic proposals per batch
		if(useImageDataset):
			stochasticOutputLearningRate = 0.0001	#default: 0.0005	#assume batchSize ~= 64	#assume batchSize ~= 1024: 0.01
			useDefaultNumLayersParam = True	#default: True
		else:
			stochasticOutputLearningRate = 0.0005	#default: 0.0005	#learning rate for dense output-layer backprop when useStochasticUpdates=True	#assume batchSize ~= 64
			useDefaultNumLayersParam = False	#default: False #disable to increase number of layers (*2)
		stochasticLayerBias = 0.0	#if bias=0 select random layer	#optional bias toward earlier layers during stochastic hidden-layer sampling	#0.0 => uniform over hidden layers; >0 => weight ~ 1/(i+1)^bias
		useDefaultNumNeuronSegmentsParam = True	#set EISANITABcontinuousVarEncodingNumBits=8
		
	if(useStochasticUpdates):
		useDynamicGeneratedHiddenConnections = False	#default: False
		#useContinuousVarEncodeMethod = "directBinary" #orig
	else:
		useDynamicGeneratedHiddenConnections = True	#default: True	#dynamically generate hidden neuron connections (else use randomly initialised hidden connections)
	if(useDynamicGeneratedHiddenConnections):
		useDynamicGeneratedHiddenConnectionsVectorised = True	#execute entire batch simultaneously
	if(useDynamicGeneratedHiddenConnections or useDynamicGeneratedCNNConnections):
		useDynamicGeneratedConnections = True
	else:
		useDynamicGeneratedConnections = False

	useInhibition = True	#default: True	#if False: only use excitatory neurons/synapses
	useConnectionWeights = True	 #mandatory: True - use sparse or dense weight tensors
	useEIneurons = False	#use separate excitatory and inhibitory neurons (else use excitatory and inhibitory connections/synapses)
	useSparseHiddenMatrix = True	#use sparse tensors to store connections (else use dense tensors)	#mandatory for any reasonably sized EISANI network
	useSparseOutputMatrix = False
	if(EISANICNNarchitectureDensePretrained):
		continuousVarMin = 0.0
		continuousVarMax = 1.0	#orig: 1.0	#max output of CNN layers is approx 1.5	#1.5
	else:
		continuousVarMin = 0.0	#sync with datasetNormaliseMinMax
		continuousVarMax = 1.0	#sync with datasetNormaliseMinMax

	numberOfSegmentsPerNeuron = 1 #number of segments per neuron
	segmentIndexToUpdate = 0 # Placeholder	#TODO: update segmentIndexToUpdate based on dataset index. Using 0 as a placeholder.

	if(useDynamicGeneratedConnections):
		targetActivationSparsityFraction = 0.1	#ideal number of neurons simultaneously active per layer

	if(useDefaultNumNeuronSegmentsParam):
		EISANITABcontinuousVarEncodingNumBits = 8	#default: 8	#number of bits to encode a continuous variable to	#for higher train performance numberNeuronSegmentsGeneratedPerSample should be increased (eg 16), however this requires a high numberNeuronSegmentsGeneratedPerSample+hiddenLayerSizeSANI to capture the larger number of input variations
		if(useDynamicGeneratedConnections):
			numberNeuronSegmentsGeneratedPerSample = 5	#default: 5	#heuristic: hiddenLayerSizeSANI//numberOfSynapsesPerSegment  	#for higher train performance numberNeuronSegmentsGeneratedPerSample should be increased substantially (eg 50), however this assigns a proportional number of additional neurons to the network (limited by hiddenLayerSizeSANI)
	else:
		EISANITABcontinuousVarEncodingNumBits = 16	#default: 16
		if(useDynamicGeneratedConnections):
			numberNeuronSegmentsGeneratedPerSample = 50
	if(useEIneurons):
		EIneuronsMatchComputation = False	#default: False	#an additional layer is required to perform the same computation as !useEIneurons
		#if(EIneuronsMatchComputation): numberNeuronSegmentsGeneratedPerSample *= 2
	if(useStochasticUpdates):
		if(useImageDataset):
			hiddenLayerSizeSANImultiplier = 1000	#default: 100
		else:
			hiddenLayerSizeSANImultiplier = 100	#default: 100	#hiddenLayerSizeSANI = hiddenLayerSize*hiddenLayerSizeSANImultiplier
	else:
		if(useDynamicGeneratedHiddenConnections):
			hiddenLayerSizeSANIbase = numberNeuronSegmentsGeneratedPerSample	#heuristic: >> hiddenLayerSizeTypical * EISANITABcontinuousVarEncodingNumBits
			initialiseSANIlayerWeightsUsingCPU = False
		else:
			hiddenLayerSizeSANI = 5120000	#default: 1280000*100 with batchSize //= numberOfLayers	#large randomly initialised sparse EISANI network width 
			initialiseSANIlayerWeightsUsingCPU = False 	#optional

	if(useDefaultSegmentSizeParam):
		numberOfSynapsesPerSegment = 5	#default: 5	#exp: 15	#number of input connections per neuron "segment"; there is 1 segment per neuron in this implementation
		segmentActivationThreshold = 3	#default: 3; allowing for 1 inhibited mismatch redundancy or 2 non inhibited mismatch redundancy	#minimum net activation required for neuron to fire (>= value), should be less than numberOfSynapsesPerSegment	#total neuron z activation expected from summation of excitatory connections to previous layer neurons
		if(not useInhibition):
			numberOfSynapsesPerSegment = numberOfSynapsesPerSegment-1
		if useDynamicGeneratedConnections: 
			useActiveBias = True	#bias positive (ceil) for odd k
	else:
		numberOfSynapsesPerSegment = 3	#default: 3
		segmentActivationThreshold = 2	#default: 2 #allowing for 1 non inhibited mismatch redundancy
		if useDynamicGeneratedConnections: 
			useActiveBias = False
			numberNeuronSegmentsGeneratedPerSample = numberNeuronSegmentsGeneratedPerSample*2
		#if !useDynamicGeneratedHiddenConnections: useDefaultNumLayersParam = False	#disable to increase number of layers
	if(useStochasticUpdates):
		segmentActivationThreshold = 0
		
	recursiveLayers = False	#default: False
	if(recursiveLayers): 
		recursiveSuperblocks = False	#default: False
		if(recursiveSuperblocks):
			recursiveSuperblocksNumber = 2
		else:
			recursiveSuperblocksNumber = 1
	else:
		recursiveSuperblocksNumber = 1
	generateConnectionsAfterPropagating = True	#default: True

if(useTabularDataset):
	datasetType = "useTabularDataset"
	EISANIcontinuousVarEncodingNumBits = EISANITABcontinuousVarEncodingNumBits
elif(useImageDataset):
	if(EISANICNNuseBinaryInput):
		datasetType = "useImageDataset"
		EISANIcontinuousVarEncodingNumBits = EISANICNNcontinuousVarEncodingNumBits
elif(useImageDataset):
	datasetType = "useNLPDataset"
	EISANIcontinuousVarEncodingNumBits = EISANINLPcontinuousVarEncodingNumBits

if(useInitOrigParam):
	useBinaryOutputConnections = True	#use binary weighted connections from hidden neurons to output neurons
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = False
	encodeDatasetBoolValuesAs1Bit = False
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	
	datasetEqualiseClassSamples = False	
	datasetEqualiseClassSamplesTest = False	
	trainNumberOfEpochs = 1 #emulate original dataset repeat x10 and epochs x10 for 4 small datasets (titanic, red-wine, breast-cancer-wisconsin, new-thyroid)
	datasetRepeatEpochModifier = '*'
	limitConnections = False
	useBinaryOutputConnectionsEffective = False
	useOutputConnectionsNormalised = False
else:
	useBinaryOutputConnections = False	#use binary weighted connections from hidden neurons to output neurons
	useDynamicGeneratedHiddenConnectionsUniquenessChecks = True
	encodeDatasetBoolValuesAs1Bit = True
	if(encodeDatasetBoolValuesAs1Bit):
		supportFieldTypeList = True
	useOutputConnectionsLastLayer = False	#use output connections only from last hidden layer to output neurons
	datasetEqualiseClassSamples = True	#default: True		#optional - advantage depends on dataset class distribution
	datasetEqualiseClassSamplesTest = False	#default: False	
	if(useStochasticUpdates):
		trainNumberOfEpochs = 10	#useStochasticUpdates applied to sparse network currently requires significantly more epochs to train
		datasetRepeatEpochModifier = 'None'	#default: 'None'
	else:
		trainNumberOfEpochs = 10
		datasetRepeatEpochModifier = '/'
	useBinaryOutputConnectionsEffective = False	
	limitConnections = False
	if(limitConnections):
		debugLimitConnectionsSequentialSANI = False	#prune every batch rather than every epoch/dataset
		debugMeasureClassExclusiveNeuronRatio = False	#measure ratio of a) class (output neuron) exclusive hidden neurons to b) non class (output neuron) exclusive hidden neurons
		debugMeasureRatioOfHiddenNeuronsWithOutputConnections = False	#measure ratio of hidden neurons with output connections to those without output connections
		printLimitConnections = True	#default: True	#print ratio of hidden neurons pruned
		limitOutputConnections = True	#optional	#prune network output connections and unused hidden neuron segments
		limitHiddenConnections = True	#optional	#prune unused hidden neuron segments and their output connections
		if(limitOutputConnections):
			#current limitation: useSparseOutputMatrix does not support individual connection pruning via limitOutputConnections:limitOutputConnectionsBasedOnPrevalence (only hidden neuron pruning)
			limitOutputConnectionsBasedOnPrevalence = True	#optional	#limit output connectivity to prevelant hidden neurons (used to prune network output connections and unused hidden neuron segments)
			limitOutputConnectionsBasedOnExclusivity = False	#experimental	#limit output connectivity to class exclusive hidden neurons (used to prune network output connections and unused hidden neuron segments)
			limitOutputConnectionsBasedOnAccuracy = True	#optional	#limit output connectivity to accurate hidden neurons; associated output class predictions observed during training (used to prune network output connections and unused hidden neuron segments)
			limitOutputConnectionsPrevalenceMin = 5	#minimum connection weight to be retained after pruning (unnormalised)
			limitOutputConnectionsAccuracyMin = 0.5	#minimum train prediction accuracy to be retained after pruning
			limitOutputConnectionsSoftmaxWeightMin = 0.5	#minimum hidden neuron normalised+softmax output connection weight to accept as predictive of output class y (ie accurate=True)
			if(limitOutputConnectionsBasedOnAccuracy):
				limitOutputConnectionsBasedOnAccuracySoftmax = True	#apply softmax after tanh norm	#more complex (especially for sparse tensors)
			if(useBinaryOutputConnections):
				useBinaryOutputConnections = False	#use integer weighted connections to calculate prevelance before prune
				useBinaryOutputConnectionsEffective = True	#after prune, output connection weights are set to 0 or 1
		if(limitHiddenConnections):
			limitHiddenConnectionsBasedOnPrevalence = True
			hiddenNeuronMinUsageThreshold = 3	#number of times a neuron must be fired across an epoch to not be pruned
	if(useBinaryOutputConnections):
		useOutputConnectionsNormalised = False
	else:
		useOutputConnectionsNormalised = True	#uses tanh to normalise output connection weights between 0 and 1
		useOutputConnectionsNormalisationRange = 1.0	#divide tanh input by useOutputConnectionsNormalisationRange
	# In stochastic mode, train dense output weights directly; disable normalisation
	if(useStochasticUpdates):
		useOutputConnectionsNormalised = False

if(useSequentialSANIactivationStrength):
	assert useBinaryOutputConnections == False

useNLPDatasetSelectTokenisation = True
useCustomLearningAlgorithm = True	#mandatory (disable all backprop optimisers)
trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/EISANIpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelEISANI'

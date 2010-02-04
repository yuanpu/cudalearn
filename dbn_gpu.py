#Copyright (c) 2009,2010 George Dahl

from dbnLayers_gpu import *

class DBN_GPU(object):
    def __init__(self, inputType, layerConstructorArgs, OutputLayerType = None):
        #at the moment we only support gaussian and binary input units
        assert(inputType.lower() in ["gaussian", "binary"])
        self.layers = []
        if inputType.lower() == "gaussian":
            self.layers.append(GaussianBinaryDBNLayer(*layerConstructorArgs[0]))
        else:
            self.layers.append(BinaryBinaryDBNLayer(*layerConstructorArgs[0]))
        for args in layerConstructorArgs[1:-1]:
            self.layers.append(BinaryBinaryDBNLayer(*args))
        if OutputLayerType == None:
            OutputLayerType = KClassCrossEntropy
        #self.layers.append(KClassCrossEntropy(*layerConstructorArgs[-1]))
        self.layers.append(OutputLayerType(*layerConstructorArgs[-1]))
        #check layer compatibility
        assert(all(self.layers[i].numVis == self.layers[i-1].numHid for i in range(1,len(self.layers))))
        assert(all(self.layers[0].mbsz == lay.mbsz for lay in self.layers))

    def packWeights(self):
        d={}
        for i,layer in enumerate(self.layers):
            for k,w in layer.packWeights().iteritems():
                assert("-" not in k)
                d[str(i)+"-"+k] = w
        return d

    def loadWeights(self, wDict, highestLayerToLoad = None):
        """
        Be careful what keys are in wDict, or we might raise an
        AssertionError. We expect keys to be annotated with the layer
        index.
        """
        if highestLayerToLoad == None:
            highestLayerToLoad = len(self.layers)
        highest = min(len(self.layers), highestLayerToLoad)
        layerDicts = [{} for i in range(len(self.layers))]
        for k in wDict:
            if not k.startswith("__"):
                assert("-" in k)
                layerId, wname = k.split("-")
                layerId = int(layerId)
                if layerId < highest:
                    layerDicts[layerId][wname] = wDict[k]
        for i in range(highest):
            self.layers[i].loadWeights(layerDicts[i])

    def loadFirstNLayers(self, wDict, n):
        assert(n<=len(self.layers))
        layerDicts = [{} for i in range(n)]
        for k in wDict:
            if not k.startswith("__"):
                assert("-" in k)
                layerId, wname = k.split("-")
                if int(layerId) < n:
                    layerDicts[int(layerId)][wname] = wDict[k]
        for i in range(n):
            self.layers[i].loadWeights(layerDicts[i])
            
    def setForAllLayers(self, name, value):
        assert(all(hasattr(layer, name) for layer in self.layers))
        for layer in self.layers:
            setattr(layer, name, value)
    
    def fpropToIth(self, inputOnGPU, i):
        """
        We propagate to the input of layer i.

        WARNING!: This function will return a CUDAMatrix or a numpy
        array depending on whether i < len(self.layers) or not!
        """
        acts = inputOnGPU 
        for lay in range(i):
            self.layers[lay].vis = acts
            acts = self.layers[lay].fprop()
        return acts

    def fprop(self, inpt):
        if isinstance(inpt, cm.CUDAMatrix):
            inputOnGPU = inpt
        else:
            inputOnGPU = cm.CUDAMatrix(reformat(inpt))
        return self.fpropToIth(inputOnGPU, len(self.layers))

    def bprop(self):
        """
        We assume outputErr has been called on the final layer
        already.
        """
        numLayers = len(self.layers)
        errSignal = self.layers[-1].errSignal
        for lay in range(numLayers):
            self.layers[numLayers-lay-1].errSignal = errSignal
            errSignal = self.layers[numLayers-lay-1].bprop()
        return errSignal

    def applySupervisedUpdates(self):
        """
        WARNING!: Only call this function right after calling bprop!
        """
        for layer in self.layers:
            layer.applySupervisedUpdates()
        
    def preTrainIth(self, freshMinibatches, epochs, i, reportMB = False, onGPU = False):
        for ep in range(epochs):
            recErr = 0
            for j,mb in enumerate(freshMinibatches()):
                if onGPU:
                    curInputsMB = mb
                else:
                    curInputsMB = cm.CUDAMatrix(reformat(mb))
                self.layers[i].step(self.fpropToIth(curInputsMB, i))
                recErr += self.layers[i].curRecErr()
                if reportMB:
                    yield (ep, j)
            yield recErr
    
    def fineTune(self, genMinibatches, epochs, loss, alsoReportErr = False):
        for ep in range(epochs):
            sumLoss, sumErr = 0, 0
            for inputVectMB, targetMB in genMinibatches():
                acts = self.fprop(inputVectMB)
                if alsoReportErr:
                    sumErr += self.layers[-1].error(targetMB)
                sumLoss += loss(acts, targetMB)
                self.layers[-1].outputErr(targetMB) #sets self.layers[-1].errSignal
                self.bprop()
                self.applySupervisedUpdates()
            if alsoReportErr:
                yield sumLoss, sumErr
            else:
                yield sumLoss
    
    def predictions(self, inp):
        mbsz = self.layers[0].mbsz
        numcases = inp.shape[1]
        numFullMinibatches = numcases / mbsz
        excess = numcases % mbsz
        pred = []
        
        for i in range(numFullMinibatches):
            idx = i*mbsz
            acts = self.fprop(inp[:,idx:idx+mbsz])

            if isinstance(acts, cm.CUDAMatrix):
                acts.copy_to_host()
                pred.append(acts.numpy_array.copy())
            else:
                pred.append(acts.copy())
        if excess != 0:
            idx = numFullMinibatches*mbsz
            mb = num.zeros((inp.shape[0], mbsz))
            mb[:,:excess] = inp[:, idx:]
            acts = self.fprop(mb)
            if isinstance(acts, cm.CUDAMatrix):
                acts.copy_to_host()
                pred.append(acts.numpy_array[:,:excess].copy())
            else:
                pred.append(acts[:,:excess].copy())
        return num.hstack(pred)
    
    def totalLoss(self, minibatches, loss = None):
        """
        Consumes an iterator over pairs of parallel input and target
        minibatches and a loss function and computes the sum loss over
        all minibatches.

        If a loss function is not provided, we use the error function
        our output layer is optimizing.
        """
        sumLoss = 0
        for inputMB, targetMB in minibatches:
            acts = self.fprop(inputMB)
            if loss == None:
                sumLoss += self.layers[-1].error(targetMB)
            else:
                sumLoss += loss(acts, targetMB)
        return sumLoss

    def classSpecificAvgLoss(self, minibatches, loss):
        numClasses = self.layers[-1].numClasses
        avgLoss = num.zeros((numClasses,))
        I = num.eye(numClasses)
        totalNumCasesPerClass = num.zeros((numClasses,))
        for inputMB, targetMB in minibatches:
            numCasesInEachClass = num.sum(targetMB, 1) #now shape (numClasses,)
            totalNumCasesPerClass += numCasesInEachClass
            acts = self.fprop(inputMB)
            for c in range(numClasses):
                if numCasesInEachClass[c] > 0:
                    #colMask = num.all(I[:,c].reshape(numClasses,1) == targetMB,0)
                    colIndices = num.nonzero(num.all(I[:,c].reshape(numClasses,1) == targetMB,0))[0]
                    avgLoss[c] += loss(acts[:,colIndices], targetMB[:,colIndices])
        assert(num.all(totalNumCasesPerClass != 0))
        avgLoss /= totalNumCasesPerClass
        return avgLoss


if __name__ == "__main__":
    print "export LD_LIBRARY_PATH=/u/gdahl/cudaLearn/"
    print "export CUDAMATDIR=/u/gdahl/cudaLearn"
    
    devId = cm.cuda_get_free_device()
    cm.cuda_set_device(devId)
    
    cm.cublas_init()
    cm.CUDAMatrix.init_random(1)
    main()
    cm.cublas_shutdown()

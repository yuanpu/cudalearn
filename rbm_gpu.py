#Copyright (c) 2009,2010 George Dahl

import numpy as num

from singleSoftmax import singleSoftmax

import cudamat as cm

class GaussianRBM(object):
    def __init__(self, numVis, numHid, mbsz = 256, initHidBias = 0.0, initWeightSigma = 0.05):
        self._mbsz = mbsz
        self.numVis, self.numHid = numVis, numHid

        #weightSigma = min(0.1, 0.1*64/num.sqrt(numHid))
        #need to decrease the weight initialization variance when we increase the number of hidden units and cdsteps
        self.visToHid = initWeightSigma*num.random.randn(numVis, numHid)
        self.visBias = num.zeros((numVis, 1))
        self.hidBias = num.zeros((numHid, 1)) + initHidBias
        
        self.init_weight_storage()
        
        self.initTemporary()

        #will be used for L1 reg and allocated at that point
        self.signVisToHid = None
        
        #set default learning parameters:
        self.setLearningParams(0.001, 0.9, 0, "L2", 1, False)

    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)
    
    def initTemporary(self):
        self.hActs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.hActProbs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.negVis = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.tempVisMB = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.negHidActProbs = None
    
    def init_weight_storage(self):
        """
        Initialize storage for gradients and gradient steps and build a list of
        weight/gradient/energy gradient/step tuples.
        """
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            self.__dict__[name] = cm.CUDAMatrix(w)
            self.__dict__["d"+name] = cm.CUDAMatrix(0.0 * w)
            
    
    def scaleDerivs(self, factor):
        """
        Scales all weight derivatives by factor (used to apply
        momentum or clear the weight derivatives).
        """
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            self.__dict__["d"+name].mult(factor)
    
    def packWeights(self):
        w_dict = {}
        for w_name in self.weightVariableNames():
            w = self.__dict__[w_name]
            w.copy_to_host()
            w_dict[w_name] = w.numpy_array
        return w_dict

    def loadWeights(self, wDict):
        for w_name in self.weightVariableNames():
            assert( wDict.has_key(w_name) )
            w = wDict[w_name]
            assert( self.__dict__[w_name].numpy_array.shape == wDict[w_name].shape )
            self.__dict__[w_name] = cm.CUDAMatrix(w)
    
    def curRecErr(self):
        self.vis.subtract(self.negVis, target = self.tempVisMB)
        return self.tempVisMB.euclid_norm()**2

    def paramNorms(self):
        d = {}
        for wname in self.weightVariableNames():
            d[wname] = self.__dict__[wname].euclid_norm() 
            d["d"+wname] = self.__dict__["d"+wname].euclid_norm()
        return d
    
    def sampleHiddens(self, hActProbsOnGPU = None):
        if hActProbsOnGPU == None:
            hActProbsOnGPU = self.hActProbs
        self.hActs.fill_with_rand()
        self.hActs.less_than(hActProbsOnGPU, target = self.hActs)
    
    def hidActProbs(self, targ = None, vis = None):
        """
        targ had better be on the gpu or None
        """
        if targ == None:
            targ = self.hActProbs
        if vis == None:
            vis = self.vis
        
        cm.dot( self.visToHid.T, vis, target = targ)
        targ.add_col_vec(self.hidBias)
        targ.apply_sigmoid()
    
    def visActProbs(self):
        cm.dot( self.visToHid, self.hActs, target = self.negVis)
        self.negVis.add_col_vec(self.visBias)
        
    def weightVariableNames(self):
        """
        Returns the names of the variables for the weights that define
        this model in a cannonical order.  The order must match up
        with the way weight derivatives get returned from CDn.
        """
        return "visToHid", "hidBias", "visBias"
    
    def CDStats(self, vis, hid, posPhase):
        """
        hid should be self.numHid by mbsz and exist on the GPU
        vis should be self.numVis by mbsz and exist on the GPU

        We modify self.d$WEIGHT_NAME as a side effect.
        """
        multiplier = 1.0 if posPhase else -1.0
        
        self.dhidBias.add_sums(hid, 1, mult = multiplier)
        self.dvisBias.add_sums(vis, 1, mult = multiplier)
        
        if posPhase:    
            self.dvisToHid.add_dot(vis, hid.T)
        else:
            self.dvisToHid.subtract_dot(vis, hid.T)
        
            
    def CDn(self):
        """
        After this function runs we will have the negative data in
        self.negVis and self.hActProbs will hold the final hidden
        activation probabilities conditioned on the negative data.
        
        This function updates the weight derivative variables.
        """
        #we depend on the following two learning parameters
        n = self.cdSteps
        momentum = self.momentum

        #@TODO: We REALLY should refactor this so CD doesn't apply the momentum itself and the step function does instead
        #apply momentum
        self.scaleDerivs(momentum)
        
        #stores hidden activation probabilities in self.hActProbs
        self.hidActProbs()
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.hActProbs, True)
        
        for i in range(n):
            #updates self.hActs
            self.sampleHiddens(self.hActProbs)
            
            #updates self.negVis
            self.visActProbs()
            
            #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
            self.hidActProbs(vis = self.negVis)

        #compute negative phase statistics and subtract them from gradient variables
        self.CDStats(self.negVis, self.hActProbs, False)
    
    def PCD(self):
        """
        After this function runs we will have the negative data in
        self.negVis and self.hActProbs will hold the final hidden
        activation probabilities conditioned on the negative data.
        
        This function updates the weight derivative variables.
        """
        self.scaleDerivs(0.0) #we use zero momentum for PCD
        #stores hidden activation probabilities in self.hActProbs
        self.hidActProbs()
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.hActProbs, True)

        if self.negHidActProbs == None:
            self.negHidActProbs = cm.CUDAMatrix(num.empty((self.numHid, self.mbsz)))
            self.negHidActProbs.assign(self.hActProbs)
        self.sampleHiddens(self.negHidActProbs)
        self.visActProbs()    
        #stores recomputed (based on self.negVis) hidden act probs in self.negHidActProbs
        self.hidActProbs(targ = self.negHidActProbs, vis = self.negVis)
        
        #compute negative phase statistics and subtract them from gradient variables
        self.CDStats(self.negVis, self.negHidActProbs, False)

    def runChain(self, sampledSteps, meanFieldSteps):
        """
        This function can be useful for reconstruction and generation
        in a DBN.  We stochastically update the hidden units
        sampleSteps times (always mean field update visibles) and then
        perform meanFieldSteps updates of the hiddens.
        """
        self.hidActProbs()
        for i in range(sampledSteps):
            #updates self.hActs
            self.sampleHiddens(self.hActProbs)
            #updates self.negVis
            self.visActProbs()
            #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
            self.hidActProbs(vis = self.negVis)
        for i in range(meanFieldSteps):
            self.hActs.assign(self.hActProbs) #this is needed since visActProbs only looks at self.hActs
            #updates self.negVis based on self.hActs
            self.visActProbs()
            #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
            self.hidActProbs(vis = self.negVis)
    
    def updateSignOfWeights(self):
        """
        We need the sign of the weights for L1 regularization.  Since
        we work on the GPU it is convenient to just allocate storage
        for these things once and periodically update the sign
        variables when the weights they depend on have changed and we
        need to know the signs.
        """
        if self.signVisToHid == None:
            self.signVisToHid = cm.CUDAMatrix(num.zeros((self.numVis, self.numHid)))
        self.visToHid.sign(target = self.signVisToHid)
        
    def decay(self):
        """
        Weight decay during pretraining.  LearningRates should be a
        dictionary with keys self.weightVariableNames() that holds the
        learning rate for each weight.
        """
        #here are the learning parameters this method depends on
        decayRate = self.weightCost
        regType = self.regType
        
        if decayRate > 0: #hopefully this saves time when decayRate == 0
            #really for L1+bias mode we should allow different weight costs for the L1 part and the bias sparsity
            assert( regType in ["L2","L1","bias","L1+bias", "L2+bias"] )
            if "L2" in regType:
                self.visToHid.mult( 1-decayRate*self.learnRate )
            if "L1" in regType:
                self.updateSignOfWeights()
                self.visToHid.subtract_mult(self.signVisToHid, decayRate*self.learnRate)
            if "bias" in regType:
                self.hidBias.add_scalar( -decayRate*self.learnRate )
    
    def step(self, data):
        """
        This function sets self.vis to point to data.
        """
        self.vis = data
        if self.doPCD:
            self.PCD()
        else:
            self.CDn()
        self.decay() #needs dictionary of learning rates, but it will reformat the rates again on its own
        for j, wname in enumerate(self.weightVariableNames()):
            self.__dict__[wname].add_mult( self.__dict__["d"+wname], self.learnRate/self.mbsz )    


    def setLearningParams(self, learnRate, momentum, weightCost, regType, cdSteps, PCD):
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightCost = weightCost
        self.regType = regType
        self.cdSteps = cdSteps
        self.doPCD = PCD
    
    def printLearningParams(self):
        print self.learnRate, self.momentum, self.regType, self.weightCost, self.doPCD, self.cdSteps
    
    
    def trainXFerEnMasse(self, inputs, numEpochs, onGPU = False):
        if onGPU:
            inputsGPU = inputs
        else:
            inputsGPU = cm.CUDAMatrix(inputs)
        
        numcases = inputsGPU.numpy_array.shape[1]
        
        num_mini_batches = numcases / self.mbsz
        batch_perm = num.random.permutation(range(num_mini_batches))
        
        for ep in range(numEpochs):
            recErr = 0
            for i in range(num_mini_batches):
                idx = batch_perm[i]
                
                self.step(inputsGPU.slice(idx*self.mbsz, (idx+1)*self.mbsz))
                recErr += self.curRecErr()
            yield recErr
    
    def train(self, freshMinibatches, numEpochs, reportMB = False):
        for ep in range(numEpochs):
            recErr = 0
            for j,mb in enumerate(freshMinibatches()):
                curInputsMB = cm.CUDAMatrix(mb)
                
                self.step(curInputsMB)
                recErr += self.curRecErr()
                #if not self.doPCD:
                #    recErr += self.curRecErr()
                if reportMB:
                    yield (ep, j)
            yield recErr
    
    def oneLayerReconstructions(self, inputs, sample = False):
        #inputs and past should be on the CPU
        hiddens = self.predictions(inputs, sample)
        recons = self.reconstructions(hiddens, False)
        return recons[:,:inputs.shape[1]]

    def predictions(self, inp, sample = False):
        
        """
        This function assumes inp resides on the cpu.  It returns a
        numpy array.

        We pad out to an integer number of minibatches.
        """
        #we return an array numHid by ceil(numcases/mbsz)
        pred = []
        
        numcases = inp.shape[1]
        numFullMinibatches = numcases / self.mbsz
        excess = numcases % self.mbsz
        
        for i in range(numFullMinibatches):
            idx = i*self.mbsz
            self.vis = cm.CUDAMatrix(inp[:,idx:idx+self.mbsz])
            
            self.hidActProbs()
            if sample:
                self.sampleHiddens(self.hActProbs)
                self.hActs.copy_to_host()
                pred.append(self.hActs.numpy_array.copy())
            else:
                self.hActProbs.copy_to_host()
                pred.append(self.hActProbs.numpy_array.copy())
        if excess != 0:
            idx = numFullMinibatches*self.mbsz
            mb = num.zeros((inp.shape[0], self.mbsz))
            mb[:,:excess] = inp[:, idx:]
            self.vis = cm.CUDAMatrix(mb)
            self.hidActProbs()
            if sample:
                self.sampleHiddens(self.hActProbs)
                self.hActs.copy_to_host()
                pred.append(self.hActs.numpy_array.copy())
            else:
                self.hActProbs.copy_to_host()
                pred.append(self.hActProbs.numpy_array.copy())
        return num.hstack(pred)
    
    def reconstructions(self, hiddens, onGPU = False):
        """
        We assume we have an integer number of
        minibatches.
        """
        #we return an array numVis by floor(numcases/self.mbsz)
        if onGPU:
            hiddensGPU = hiddens
        else:
            hiddensGPU = cm.CUDAMatrix(hiddens)

        numcases = hiddensGPU.numpy_array.shape[1]
        num_mini_batches = numcases / self.mbsz

        recons = []
        for i in range(num_mini_batches):
            
            self.hActs = hiddensGPU.slice(i*self.mbsz, (i+1)*self.mbsz)
            self.visActProbs()
            self.negVis.copy_to_host()
            recons.append(self.negVis.numpy_array.copy())
        self.initTemporary()
        return num.hstack(recons)
    


class BinaryRBM(GaussianRBM):
    def visActProbs(self):
        GaussianRBM.visActProbs(self)
        self.negVis.apply_sigmoid()

class SoftmaxRBM(GaussianRBM):
    def __init__(self, numVis, numHid, k, mbsz = 256, initHidBias = 0.0, initWeightSigma = 0.05):
        assert(numVis % k == 0)
        self.k = k
        GaussianRBM.__init__(self,  numVis, numHid, mbsz, initHidBias, initWeightSigma)

    def initTemporary(self):
        self.hActs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.hActProbs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.negVis = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.tempVisMB = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.negHidActProbs = None
        self.tempRow = cm.CUDAMatrix(num.zeros((1,(self.numVis/self.k)*self.mbsz)))
    
    def visActProbs(self):
        GaussianRBM.visActProbs(self) #now self.negVis holds the net input to the visible units
        #self.negVis has shape (self.numVis, self.mbsz)
        self.negVis.reshape((self.k, self.mbsz*self.numVis/self.k))
        singleSoftmax(self.negVis, self.tempRow)
        self.negVis.reshape((self.numVis, self.mbsz))

def gpu_batches(data, past, mbs, transpose = True):
    """
    We assume that the first dimension of the data is the number of cases.

    We generate minibatches of data and delayed data of the appropriate size transposed for use on the GPU.

    If we can't fill the last minibatch, we discard that data.
    """
    numCases, numDims = data.shape
    numBatches = numCases/mbs
    for i in range(numBatches):
        if transpose:
            yield (data[i*mbs:(i+1)*mbs,:].transpose(), [p[i*mbs:(i+1)*mbs,:].transpose() for p in past])
        else:
            yield (data[i*mbs:(i+1)*mbs,:], [p[i*mbs:(i+1)*mbs,:] for p in past])

def main():
    pass

if __name__ == "__main__":
    print "export LD_LIBRARY_PATH=/u/gdahl/cudaLearn/"
    print "export CUDAMATDIR=/u/gdahl/cudaLearn"
    
    devId = cm.cuda_get_free_device()
    cm.cuda_set_device(devId)
    
    cm.cublas_init()
    cm.CUDAMatrix.init_random(1)
    main()
    cm.cublas_shutdown()


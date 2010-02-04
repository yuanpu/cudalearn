#Copyright (c) 2009,2010 George Dahl

import numpy as num

import cudamat as cm
from cudamat import reformat
from singleSoftmax import maskedSingleSoftmax

def getFilteringDist(net, data, index, preSigmoid = False):
    """
    We use this name to correspond more closely to Graham's matlab
    code.  This function sends the visible data stored in data through
    net to produce hidden unit activations for every valid position of
    net.  The valid positions are given by index.
    """
    assert(len(index.shape)==1)
    pred = []
    
    numcases = index.shape[0]
    num_mini_batches = numcases / net.mbsz
    excess = numcases - num_mini_batches*net.mbsz
    
    for mb in range(num_mini_batches):
        mbIdx = index[ mb*net.mbsz:(mb+1)*net.mbsz ]
        net.vis = cm.CUDAMatrix(reformat(data[:,mbIdx]))
        net.past = [ cm.CUDAMatrix(reformat(data[:,mbIdx-i-1])) for i in range(net.numPrev) ]

        if preSigmoid:
            net.hidNetInpts()
        else:
            net.hidActProbs()
        net.hActProbs.copy_to_host()
        pred.append(net.hActProbs.numpy_array.copy())
    if excess > 0:
        batch = num.zeros(net.vis.shape)
        mbIdx = index[ num_mini_batches*net.mbsz:]
        batch[:,:excess] = data[:,mbIdx]
        net.vis = cm.CUDAMatrix(reformat(batch))
        net.past = []
        for i in range(net.numPrev):
            batch[:,:excess] = data[:,mbIdx-i-1]
            net.past.append(cm.CUDAMatrix(reformat(batch)))
        if preSigmoid:
            net.hidNetInpts()
        else:
            net.hidActProbs()
        net.hActProbs.copy_to_host()
        pred.append(net.hActProbs.numpy_array.copy()[:,:excess])
            
    return num.hstack(pred)

class GaussianCRBM(object):
    def __init__(self, numVis, numHid, prevFrames, initHidBias = 0.0):
        self.numVis, self.numHid, self.numPrev = numVis, numHid, prevFrames
        self._mbsz = 256
        
        self.visToHid = 0.1*num.random.randn(numVis, numHid)
        self.visBias = num.zeros((numVis, 1))
        self.hidBias = num.zeros((numHid, 1)) + initHidBias
        
        #self.A[0] and self.B[0] are the weights from the most recent frame
        self.A = [0.01*num.random.randn(numVis, numVis) for i in range(self.numPrev)]
        self.B = [0.01*num.random.randn(numVis, numHid)  for i in range(self.numPrev)]
        
        self.init_weight_storage()
        
        self.initTemporary()

        #will be used for L1 reg and allocated at that point
        self.signVisToHid = None
        self.signA = None
        self.signB = None
        
        #set default learning parameters:
        self.setLearningParams()
        
        #total GPU storage costs excluding input data (self.past and self.vis):
        # 2W + 2*numHid*mbsz+2*numVis*mbsz
        #where W is the total space cost of all the weights of the model

    def setLearningParams(self, learnRate = 0.001, momentum = 0.9, weightCost = 0, regType = "L2", cdSteps = 1, \
                          pastNoise = 0, arWeightCost = None):
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightCost = weightCost
        self.regType = regType
        self.cdSteps = cdSteps
        self.pastNoise = pastNoise
        self.arWeightCost = arWeightCost
    
    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)
    
    def initTemporary(self):
        self.hActs = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.hActProbs = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.negVis = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        self.tempVisMB = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        self.dynamicHidBias = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.dynamicVisBias = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
    
    def init_weight_storage(self):
        """
        Initialize storage for gradients and gradient steps and build a list of
        weight/gradient/energy gradient/step tuples.
        """
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            if not isinstance(w, list):
                self.__dict__[name] = cm.CUDAMatrix(reformat(w))
                self.__dict__["d"+name] = cm.CUDAMatrix(reformat(0.0 * w))
            else:
                self.__dict__[name] = [cm.CUDAMatrix(reformat(x)) for x in w]
                self.__dict__["d"+name] = [cm.CUDAMatrix(reformat(0.0*part)) for part in w]
    
    def scaleDerivs(self, factor):
        """
        Scales all weight derivatives by factor (used to apply
        momentum or clear the weight derivatives).
        """
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            if not isinstance(w, list):
                self.__dict__["d"+name].mult_by_scalar(factor)
            else:
                for i in range(len(w)):
                    self.__dict__["d"+name][i].mult_by_scalar(factor)
    
    def pack_weights(self):
        w_dict = {}
        for w_name in self.weightVariableNames():
            w = self.__dict__[w_name]
            if isinstance(w, list):
                for part in w:
                    part.copy_to_host()
                w_dict[w_name] = [part.numpy_array for part in w]
            else:
                w.copy_to_host()
                w_dict[w_name] = w.numpy_array
        return w_dict

    def loadWeights(self, wDict):
        """
        This code is terrible.
        """
        assert(all(wName in wDict for wName in self.weightVariableNames()))
        for w_name in wDict:
            if w_name in self.weightVariableNames():
                w = wDict[w_name]
                if isinstance(w, list) or w_name in ["A","B"]:
                    assert( all(self.__dict__[w_name][i].numpy_array.shape == wDict[w_name][i].shape for i in range(len(wDict[w_name])) ) )
                    self.__dict__[w_name] = [cm.CUDAMatrix(reformat(part)) for part in w]
                else:
                    assert( self.__dict__[w_name].numpy_array.shape == wDict[w_name].shape )
                    self.__dict__[w_name] = cm.CUDAMatrix(reformat(w))
    
    def curRecErr(self):
        self.vis.subtract(self.negVis, target = self.tempVisMB)
        return self.tempVisMB.euclid_norm()**2
    
    def allWeightsMatlabFormat(self):
        weights = self.pack_weights()
        d = {}
        d["w"] = weights["visToHid"].transpose()
        d["bi"] = weights["visBias"]
        d["bj"] = weights["hidBias"]

        #this chunk of code depends on scipy version >= 8 so savemat works right
        d["A"] = num.empty( (self.numVis, self.numVis, self.numPrev) )
        d["B"] = num.empty( (self.numHid, self.numVis, self.numPrev) )
        for i in range(self.numPrev):
            d["A"][:,:,i] = weights["A"][i].transpose()
            d["B"][:,:,i] = weights["B"][i].transpose()

        
        #for i in range(self.numPrev):
        #    d["A%d" % i] = weights["A"][i].transpose()
        #    d["B%d" % i] = weights["B"][i].transpose()

        return d
    
    def sampleHiddens(self, hActProbsOnGPU = None):
        if hActProbsOnGPU == None:
            hActProbsOnGPU = self.hActProbs
        self.hActs.fill_with_rand()
        self.hActs.less_than(hActProbsOnGPU, target = self.hActs)

    def hidNetInpts(self, recomputeDynamicBias = True, targ = None, vis = None):
        """
        targ had better be on the gpu or None
        """
        if recomputeDynamicBias:
            self.updateDynamicHidBias()

        if targ == None:
            targ = self.hActProbs
        if vis == None:
            vis = self.vis
        
        cm.dot( self.visToHid.T, vis, target = targ)
        targ.add(self.dynamicHidBias)
        targ.add_col_vec(self.hidBias)

    def hidActProbs(self, recomputeDynamicBias = True, targ = None, vis = None):
        """
        targ had better be on the gpu or None
        """
        if targ == None:
            targ = self.hActProbs
        self.hidNetInpts(recomputeDynamicBias, targ, vis)
        targ.apply_sigmoid()
    
    def updateDynamicHidBias(self):
        self.dynamicHidBias.mult_by_scalar(0.0)
        for i in range(len(self.B)):
            #self.past[i] is the (i+1)-steps delayed frame of history
            #self.past[i] is numVis by mbsz
            #self.B[i] is numVis by numHid
            self.dynamicHidBias.add_dot(self.B[i].T, self.past[i])
        
    def updateDynamicVisBias(self):
        self.dynamicVisBias.mult_by_scalar(0.0)
        for i in range(len(self.A)):
            self.dynamicVisBias.add_dot(self.A[i].T, self.past[i])
    
    def visActProbs(self, recomputeDynamicBias):
        
        if recomputeDynamicBias:
            self.updateDynamicVisBias()
        
        cm.dot( self.visToHid, self.hActs, target = self.negVis)
        self.negVis.add(self.dynamicVisBias)
        self.negVis.add_col_vec(self.visBias)
        
    def weightVariableNames(self):
        """
        Returns the names of the variables for the weights that define
        this model in a cannonical order.  The order must match up
        with the way weight derivatives get returned from CDn.
        """
        return "visToHid", "hidBias", "visBias", "A", "B"
    
    def CDStats(self, vis, past, hid, posPhase):
        """
        hid should be self.numHid by mbsz and exist on the GPU
        vis should be self.numVis by mbsz and exist on the GPU
        past should be a length self.numPrev list of variables like vis

        This function depends on self.dynamicVisBias being up to date!!

        We modify self.d$WEIGHT_NAME as a side effect and clobber self.tempVisMB.
        """
        vis.subtract(self.dynamicVisBias, target = self.tempVisMB)
        self.tempVisMB.add_col_mult(self.visBias, -1.0) #so we are subtracting, not adding

        multiplier = 1.0 if posPhase else -1.0
        
        self.dhidBias.add_sums(hid, 1, mult = multiplier)
        self.dvisBias.add_sums(vis, 1, mult = multiplier)
        
        if posPhase:    
            self.dvisToHid.add_dot(vis, hid.T)
            
            for i in range(self.numPrev):
                self.dA[i].add_dot( past[i], self.tempVisMB.T )
                self.dB[i].add_dot( past[i], hid.T )
        else:
            self.dvisToHid.subtract_dot(vis, hid.T)
            
            for i in range(self.numPrev):
                self.dA[i].subtract_dot( past[i], self.tempVisMB.T )
                self.dB[i].subtract_dot( past[i], hid.T )
        
            
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

        #apply momentum
        self.scaleDerivs(momentum)
        
        #stores hidden activation probabilities in self.hActProbs and sets dynamic hidden biases 
        self.hidActProbs()

        self.updateDynamicVisBias() #CDStats depends on self.dynamicVisBias being correct
        
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.past, self.hActProbs, True)
        
        
        for i in range(n):
            #updates self.hActs
            self.sampleHiddens(self.hActProbs)
            
            #updates self.negVis and if i == 0 computes self.dynamicVisBias
            self.visActProbs(False) #no need to recompute self.dynamicVisBias
            
            #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
            self.hidActProbs(False, vis = self.negVis)

        #compute negative phase statistics and subtract them from gradient variables
        self.CDStats(self.negVis, self.past, self.hActProbs, False)
        
    
    def reformatLearningRates(self, learnRate):
        if isinstance(learnRate, dict):
            assert( all(name in learnRate for name in self.weightVariableNames() ) )
            return learnRate
        rates = {}
        assert( type(learnRate) == float or type(learnRate) == int )
        for name in self.weightVariableNames():
            rates[name] = learnRate
        return rates

    def updateSignOfWeights(self):
        """
        We need the sign of the weights for L1 regularization.  Since
        we work on the GPU it is convenient to just allocate storage
        for these things once and periodically update the sign
        variables when the weights they depend on have changed and we
        need to know the signs.
        """
        if self.signVisToHid == None or self.signA == None or self.signB == None:
            self.signVisToHid = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.numHid))))
            self.signA = [cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.numVis)))) for i in range(self.numPrev)]
            self.signB = [cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.numHid)))) for i in range(self.numPrev)]
        self.visToHid.sign(target = self.signVisToHid)
        for i in range(self.numPrev):
            self.A[i].sign(target = self.signA[i])
            self.B[i].sign(target = self.signB[i])        
        
    def decay(self):
        """
        Weight decay during pretraining.  LearningRates should be a
        dictionary with keys self.weightVariableNames() that holds the
        learning rate for each weight.
        """
        #here are the learning parameters this method depends on
        decayRate = self.weightCost
        arDecayRate = self.arWeightCost if self.arWeightCost != None else decayRate
        learningRates = self.reformatLearningRates(self.learnRate) # we reformat in case self.learnRate isn't a dict
        regType = self.regType
        
        if decayRate > 0: #hopefully this saves time when decayRate == 0
            #really for L1+bias mode we should allow different weight costs for the L1 part and the bias sparsity
            assert( regType in ["L2","L1","bias","L1+bias", "L2+bias"] )
            if "L2" in regType:
                self.visToHid.mult_by_scalar( 1-decayRate*learningRates['visToHid'] )
                for i in range(self.numPrev):
                    self.A[i].mult_by_scalar( 1-arDecayRate*learningRates['A'] )
                    self.B[i].mult_by_scalar( 1-arDecayRate*learningRates['B'] )
            if "L1" in regType:
                self.updateSignOfWeights()
                self.visToHid.subtract_mult(self.signVisToHid, decayRate*learningRates['visToHid'])
                for i in range(self.numPrev):
                    self.A[i].subtract_mult(self.signA[i], arDecayRate*learningRates['A'])
                    self.B[i].subtract_mult(self.signB[i], arDecayRate*learningRates['B'])
            if "bias" in regType:
                self.hidBias.add_scalar( -decayRate*learningRates['hidBias'] )
    
    def step(self, data, past):
        """
        This function sets references in self.vis and self.past to
        point to data and past.
        """
        self.vis = data
        self.past = past
        self.CDn()
        rates = self.reformatLearningRates(self.learnRate)
        self.decay() #needs dictionary of learning rates, but it will reformat the rates again on its own
        for j, wname in enumerate(self.weightVariableNames()):
            if type(self.__dict__[wname]) == list:
                for i in range(self.numPrev):
                    self.__dict__[wname][i].add_mult( self.__dict__["d"+wname][i], rates[wname]/self.mbsz )
            else: #we assume it is a numpy array
                self.__dict__[wname].add_mult( self.__dict__["d"+wname], rates[wname]/self.mbsz )    

    def trainLowMemory(self, data, index, numEpochs, reportMB = False):
        assert(data.dtype == num.dtype('float32'))
        numcases = len(index)
        
        num_mini_batches = numcases / self.mbsz
        indexPerm = num.random.permutation(range(numcases))

        noise = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        for ep in range(numEpochs):
            recErr = 0
            for mb in range(num_mini_batches):
                mbIndex = index[ indexPerm[mb*self.mbsz:(mb+1)*self.mbsz] ]

                curInputsMB_CPU = data[:, mbIndex]
                curPastMB_CPU = [data[:, mbIndex-i-1] for i in range(self.numPrev)]
                curInputsMB = cm.CUDAMatrix(reformat(curInputsMB_CPU))
                curPastMB = [cm.CUDAMatrix(reformat(p)) for p in curPastMB_CPU]
                if self.pastNoise > 0:
                    for i in range(self.numPrev):
                        noise.fill_with_randn()
                        curPastMB[i].add_mult(noise, self.pastNoise)
                
                self.step(curInputsMB, curPastMB)
                recErr += self.curRecErr()
                if reportMB:
                    yield (mb, num_mini_batches)
            yield recErr
    
    def oneLayerReconstructions(self, inputs, past, sample = False):
        #inputs and past should be on the CPU
        hiddens = self.predictions(inputs, past, sample)
        recons = self.reconstructions(past, hiddens, False)
        return recons[:,:inputs.shape[1]]

    def predictions(self, inp, past, sample = False):
        
        """
        This function assumes inp and past reside on the cpu.  It
        returns a numpy array.

        We assume an integer number of minibatches and any cases
        beyond mbsz*floor(numcases/mbsz) are ignored.
        """
        #we return an array numHid by floor(numcases/mbsz)
        pred = []
        
        numcases = inp.shape[1]
        num_mini_batches = numcases / self.mbsz
        
        for i in range(num_mini_batches):
            idx = i*self.mbsz
            self.vis = cm.CUDAMatrix(reformat(inp[:,idx:idx+self.mbsz]))
            self.past = [ cm.CUDAMatrix(reformat(p[:,idx:idx+self.mbsz])) for p in past ]
            
            self.hidActProbs()
            if sample:
                self.sampleHiddens(self.hActProbs)
                self.hActs.copy_to_host()
                pred.append(self.hActs.numpy_array.copy())
            else:
                self.hActProbs.copy_to_host()
                pred.append(self.hActProbs.numpy_array.copy())
        return num.hstack(pred)
    
    def reconstructions(self, past, hiddens, onGPU = False):
        """
        We assume we have an integer number of
        minibatches.
        """
        #we return an array numVis by floor(numcases/mbsz)
        if onGPU:
            pastGPU = past
            hiddensGPU = hiddens
        else:
            pastGPU = [cm.CUDAMatrix(reformat(p)) for p in past]
            hiddensGPU = cm.CUDAMatrix(reformat(hiddens))

        numcases = hiddensGPU.numpy_array.shape[1]
        num_mini_batches = numcases / self.mbsz

        recons = []
        for i in range(num_mini_batches):
            self.past = [p.slice(i*self.mbsz, (i+1)*self.mbsz) for p in pastGPU]
            self.hActs = hiddensGPU.slice(i*self.mbsz, (i+1)*self.mbsz)
            self.visActProbs(True)
            self.negVis.copy_to_host()
            recons.append(self.negVis.numpy_array.copy())

        return num.hstack(recons)
    


def padToMinibatch(matrixOnCPU, mbsz):
    if matrixOnCPU.shape[1] % mbsz == 0:
        return matrixOnCPU, 0
    pad_num = mbsz - matrixOnCPU.shape[1] % mbsz
    return num.hstack( (matrixOnCPU, num.zeros((matrixOnCPU.shape[0], pad_num))) ), pad_num

class BinaryCRBM(GaussianCRBM):
    def visActProbs(self, recomputeDynamicBias):
        GaussianCRBM.visActProbs(self, recomputeDynamicBias)
        self.negVis.apply_sigmoid()

    def setLearningParams(self, learnRate = 0.04, momentum = 0.9, weightCost = 0, regType = "L2", cdSteps = 1, \
                          pastNoise = 0, arWeightCost = None, samplePast = False):
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightCost = weightCost
        self.regType = regType
        self.cdSteps = cdSteps
        self.pastNoise = pastNoise
        self.samplePast = samplePast
        self.arWeightCost = arWeightCost
        
    def trainLowMemory(self, data, index, numEpochs, reportMB = False):
        assert(data.dtype == num.dtype('float32'))
        numcases = len(index)
        
        num_mini_batches = numcases / self.mbsz
        indexPerm = num.random.permutation(range(numcases))

        noise = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        noiseThresh = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        noiseThresh.assign_scalar(1.0-self.pastNoise)
        for ep in range(numEpochs):
            recErr = 0
            for mb in range(num_mini_batches):
                mbIndex = index[ indexPerm[mb*self.mbsz:(mb+1)*self.mbsz] ]
                
                curInputsMB_CPU = data[:, mbIndex]
                curPastMB_CPU = [data[:, mbIndex-i-1] for i in range(self.numPrev)]
                curInputsMB = cm.CUDAMatrix(reformat(curInputsMB_CPU))
                curPastMB = [cm.CUDAMatrix(reformat(p)) for p in curPastMB_CPU]
                for i in range(self.numPrev):
                    if self.pastNoise > 0 and not self.samplePast:
                        noise.fill_with_rand()
                        noise.less_than(noiseThresh, target = noise)
                        curPastMB[i].mult(noise)
                    if self.samplePast:
                        noise.fill_with_rand()
                        noise.less_than(curPastMB[i], target = curPastMB[i])
                
                self.step(curInputsMB, curPastMB)
                recErr += self.curRecErr()
                if reportMB:
                    yield (mb, num_mini_batches)
            yield recErr

class HybridCRBM(GaussianCRBM):
    """
    This class implements a hybrid crbm with a single softmax unit and
    some gaussian units for the visible units.
    """
    def __init__(self, numVis, numHid, prevFrames, smsz, initHidBias = 0.0):
        assert(0 <= smsz <= numVis)
        self.smsz = smsz
        GaussianCRBM.__init__(self, numVis, numHid, prevFrames, initHidBias)
        
    def initTemporary(self):
        self.hActs = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.hActProbs = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.negVis = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        self.tempVisMB = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        self.dynamicHidBias = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.dynamicVisBias = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))

        self.sMask = num.zeros((self.numVis, self.mbsz))
        self.sMask[:self.smsz,:] = 1
        self.gaussMask = 1-self.sMask
        
        self.onesCol = cm.CUDAMatrix(reformat(num.ones((self.numVis,1))))
        self.sMask = cm.CUDAMatrix(reformat(self.sMask))
        self.gaussMask = cm.CUDAMatrix(reformat(self.gaussMask))
        self.tempRow = cm.CUDAMatrix(reformat(num.zeros((1, self.mbsz))))
        #self.tempBinVisMB = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        
    def setLearningParams(self, learnRate = 0.001, momentum = 0.9, weightCost = 0, regType = "L2", cdSteps = 1, \
                          pastNoise = 0, arWeightCost = None, pastNoiseSM = 0):
        GaussianCRBM.setLearningParams(self, learnRate, momentum, weightCost, regType, cdSteps, pastNoise, arWeightCost)
        self.pastNoiseSM = pastNoiseSM
    
    def visActProbs(self, recomputeDynamicBias):
        GaussianCRBM.visActProbs(self, recomputeDynamicBias)
        maskedSingleSoftmax(self.negVis, self.tempVisMB, self.sMask, self.gaussMask, self.onesCol, self.tempRow)

    def trainLowMemory(self, data, index, numEpochs, reportMB = False):
        assert(data.dtype == num.dtype('float32'))
        numcases = len(index)
        
        num_mini_batches = numcases / self.mbsz
        indexPerm = num.random.permutation(range(numcases))
        
        noise = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        for ep in range(numEpochs):
            recErr = 0
            for mb in range(num_mini_batches):
                mbIndex = index[ indexPerm[mb*self.mbsz:(mb+1)*self.mbsz] ]
                
                curInputsMB_CPU = data[:, mbIndex]
                curPastMB_CPU = [data[:, mbIndex-i-1] for i in range(self.numPrev)]
                if self.pastNoiseSM > 0:
                    for i in range(self.numPrev):
                        smNoise = (self.pastNoiseSM/self.smsz)*num.random.rand(self.smsz, self.mbsz)
                        #smNoise[0,:] = 0
                        #smNoise /= self.smsz-1
                        curPastMB_CPU[i][:self.smsz,:] = (curPastMB_CPU[i][:self.smsz,:] + smNoise)/(1+self.pastNoiseSM)
                        
                curInputsMB = cm.CUDAMatrix(reformat(curInputsMB_CPU))
                curPastMB = [cm.CUDAMatrix(reformat(p)) for p in curPastMB_CPU]
                if self.pastNoise > 0:
                    for i in range(self.numPrev):
                        noise.fill_with_randn()
                        noise.mult(self.gaussMask)
                        curPastMB[i].add_mult(noise, self.pastNoise)
                
                self.step(curInputsMB, curPastMB)
                recErr += self.curRecErr()
                if reportMB:
                    yield (mb, num_mini_batches)
            yield recErr


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



def main1():
    net = BinaryCRBM(10,16,2)
    data = loadmat("brazilRainfall.mat")["batchdata"]
    chunks = [(data[i*90+2:(i+1)*90,:], [data[i*90+1:(i+1)*90-1,:], data[i*90:(i+1)*90-2,:]]) for i in range(24)]
    
    data = num.vstack( [c[0] for c in chunks] )
    past = [ num.vstack( [c[1][i] for c in chunks] ) for i in range(2)]

    data = data.transpose()
    past = [p.transpose() for p in past]

    print data.shape
    print data.shape[1]/64
    for p in past:
        print p.shape

    net.learnRate = 0.002
    net.momentum = 0.9
    net.weightCost = 0
    for j,err in enumerate(net.trainXFerEnMasse(data, past, 100)):
        print j+1, err
    

    ex = cm.CUDAMatrix(reformat(num.array([[1,1],[2,3]])))
    print ex.euclid_norm()
    

def main2():
    pass

from scipy.io import loadmat

if __name__ == "__main__":
    print "export LD_LIBRARY_PATH=/u/gdahl/cudaLearn/"
    print "export CUDAMATDIR=/u/gdahl/cudaLearn"
    
    devId = cm.cuda_get_free_device()
    cm.cuda_set_device(devId)
    
    cm.cublas_init()
    cm.CUDAMatrix.init_random(1)
    main1()
    cm.cublas_shutdown()


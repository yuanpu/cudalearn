#Copyright (c) 2009,2010 George Dahl

import numpy as num

import cudamat as cm
from cudamat import reformat

from scipy.io import loadmat, savemat

def logOnePlusExp(x, temp, targ = None):
    """
    When this function is done, x should contain log(1+exp(x)).  We
    clobber the value of temp.  We compute log(1+exp(x)) as x +
    log(1+exp(-x)), which will hopefully be more finite-precision
    friendly.
    """
    assert(x.shape == temp.shape)
    x.mult(-1, target = temp)
    cm.exp(temp)
    temp.add(1)
    cm.log(temp)
    x.add(temp, target = targ)
    #temp.assign(x)
    #cm.exp(temp)
    #temp.add_scalar(1)
    #cm.log(temp)
    #if targ == None:
    #    x.assign(temp)
    #else:
    #    targ.assign(temp)

def negate(x):
    """
    Replace x with 1-x.
    """
    x.mult(-1)
    x.add(1)

def zeroOutPositives(matrix, tempMatrix):
    matrix.less_than(0, target = tempMatrix)
    #now tempMatrix holds a 1 for negative entries and 0 for non-negative entries
    matrix.mult(tempMatrix)

def columnNorms(mat, tempMat, result):
    assert(mat.shape == tempMat.shape)
    assert(result.shape == (1, mat.shape[1]))
    #cm.pow(mat, 2, target = tempMat)
    mat.mult(mat, target = tempMat)
    tempMat.sum(axis = 0, target = result)
    cm.sqrt(result)

small = 0.001

def normalizeInputData(vis, tempVis, sqColLens, normalizer, normalizedVis):
    """
    Our input is vis and our outputs are sqColLens, normalizer, and
    normalizedVis.  We clobber tempVis.
    """
    numVis, mbsz = vis.shape
    assert(sqColLens.shape == (1, mbsz))
    assert(sqColLens.shape == normalizer.shape)
    assert(tempVis.shape == vis.shape == normalizedVis.shape)

    vis.mult(vis, target = tempVis)
    tempVis.sum(axis = 0, target = sqColLens)
    sqColLens.mult(1.0/numVis, target = normalizer)
    normalizer.add(small)
    cm.sqrt(normalizer)
    normalizer.reciprocal()
    vis.mult_by_row(normalizer, target = normalizedVis)


class CovGRBM(object):
    """
    Warning!  This class should only be used on PCA whitened data!

    Also, this model has no visible bias, which is another reason it
    isn't appropriate for higher layers.
    """
    def __init__(self, numVis, numFact, numHid, mbsz = 256, initWeightSigma = 0.05):
        self._mbsz = mbsz
        self.numVis = numVis
        self.numFact = numFact
        self.numHid = numHid

        #@TODO: remove this later
        #num.random.seed(8)
        self.visToFact = cm.CUDAMatrix(initWeightSigma*num.random.randn(numVis, numFact))
        self.dvisToFact = cm.CUDAMatrix(num.zeros((numVis, numFact)))
        self.randomSparseFactToHid() #creates self.factToHid
        self.dfactToHid = cm.CUDAMatrix(num.zeros(self.factToHid.shape))
        #self.hidBias = cm.CUDAMatrix(2*num.ones((numHid, 1))) #initialize with positive bias
        self.hidBias = cm.CUDAMatrix(1.5*num.ones((numHid, 1))) #initialize with positive bias
        self.dhidBias = cm.CUDAMatrix(num.zeros((numHid, 1)))
        self.visBias = cm.CUDAMatrix(num.zeros((numVis, 1)))
        self.dvisBias = cm.CUDAMatrix(num.zeros((numVis, 1)))
        
        self.signVisToFact = None
        self.signFactToHid = None

        self.factToHidMask = None

        self.normVisToFact = 1.0
        
        self.initTemporary()
        
        self.visToFactColNorms = cm.CUDAMatrix(num.ones((1, numFact)))
        columnNorms(self.visToFact, self.tempVisToFact, self.visToFactColNorms)

        self.curVisToFactColNorms = cm.CUDAMatrix(num.ones((1, numFact)))
        columnNorms(self.visToFact, self.tempVisToFact, self.curVisToFactColNorms)
        
        self.setLearningParams()

    def packWeights(self):
        w_dict = {}
        for w_name in self.weightVariableNames():
            w = self.__dict__[w_name]
            w_dict[w_name] = w.asarray()
        return w_dict
    
    def loadWeights(self, wDict):
        for w_name in wDict:
            if not w_name.startswith("__"):
                if self.__dict__.has_key(w_name):
                    w = wDict[w_name]
                    assert( self.__dict__[w_name].shape == w.shape )
                    self.__dict__[w_name] = cm.CUDAMatrix(w)
                else:
                    print w_name, "not found in mcRBM, skipping"
        self.initTemporary()
    
    def setLearningParams(self, **kwargs):
        self.stepSizeIsMean = True

        self.renormStartEpoch = -1

        self.maxColNorm = None #if this isn't None, we override self.allColsSame
        self.allColsSame = True
        
        self.hmcSteps = 20
        self.hmcStepSize = 0.01
        self.targetRejRate = 0.1
        self.runningAvRej = self.targetRejRate
        self.maxStepSize = 0.25
        self.minStepSize = 0.001
        
        self.learnRateVF = 0.02
        self.learnRateFH = 0.002
        self.learnRateHB = 0.01
        self.learnRateVB = 0.01
        
        self.momentum = 0
        
        self.weightCost = 0.001
        self.regType = "L1"
        
        params = set(["hmcSteps", "hmcStepSize", "targetRejRate", "maxStepSize", \
                      "minStepSize", "learnRateVF", "learnRateFH", "learnRateHB",\
                      "learnRateVB", "weightCost", "regType"])
        
        for k in params:
            if k in kwargs:
                self.__dict__[k] = kwargs[k]
        
        
    def weightVariableNames(self):
        """
        Returns the names of the variables for the weights that define
        this model in a cannonical order.
        """
        return "visToFact", "factToHid", "hidBias", "visBias"
    
    def initTemporary(self):
        self.factResponses = cm.CUDAMatrix(num.zeros((self.numFact, self.mbsz)))
        self.factResponsesSq = cm.CUDAMatrix(num.zeros((self.numFact, self.mbsz)))
        self.hActs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.hActProbs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.hNetInputs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.negVis = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.tempVisMB = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.normalizedVisMB = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.tempFactMB = cm.CUDAMatrix(num.zeros((self.numFact, self.mbsz)))
        self.tempHidMB = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.vel = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))

        self.sqColLens = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.tempHidRow = cm.CUDAMatrix(num.zeros((1, self.numHid)))
        self.tempFactToHid = cm.CUDAMatrix(num.zeros(self.factToHid.shape))
        self.tempVisToFact = cm.CUDAMatrix(num.zeros(self.visToFact.shape))
        self.tempFactRow = cm.CUDAMatrix(num.zeros((1, self.numFact)))
        
        #this variable could be eliminated and we could reuse prevHamil
        self.thresh = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.tempRow = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.tempRow2 = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.tempRow3 = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.prevHamil = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.hamil = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.accel = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.normalizedAccel = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))

        self.tempScalar = cm.CUDAMatrix(num.zeros((1,1)))

    def updateSignOfWeights(self):
        """
        We need the sign of the weights for L1 regularization.  Since
        we work on the GPU it is convenient to just allocate storage
        for these things once and periodically update the sign
        variables when the weights they depend on have changed and we
        need to know the signs.
        """
        if self.signVisToFact == None:
            self.signVisToFact = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.numFact))))
        if self.signFactToHid == None: #probably not really needed since we constrain it to be negative
            self.signFactToHid = cm.CUDAMatrix(reformat(num.zeros((self.numFact, self.numHid))))
        self.visToFact.sign(target = self.signVisToFact)
        self.factToHid.sign(target = self.signFactToHid)
    
    def decay(self):
        #here are the learning parameters this method depends on
        decayRate = self.weightCost
        regType = self.regType
        
        if decayRate > 0: #hopefully this saves time when decayRate == 0
            #at the moment I don't feel like having L2 weight decay as an option
            assert( regType in ["L1"] )
            #assert( regType in ["L2","L1"] )
            #if "L2" in regType:
            #    self.visToFact.mult( 1-decayRate*self.learnRate )
            #    #it doesn't really make sense to use L2 on factToHid since we keep the columns at a constant norm
            if "L1" in regType:
                self.updateSignOfWeights()
                #self.visToFact.subtract_mult(self.signVisToFact, decayRate*self.learnRateVF)
                self.dvisToFact.subtract_mult(self.signVisToFact, decayRate)
                #self.factToHid.subtract_mult(self.signFactToHid, decayRate*self.learnRateFH)
                self.dfactToHid.subtract_mult(self.signFactToHid, decayRate)
            
    def scaleDerivs(self, factor):
        """
        Scales all weight derivatives by factor (used to apply
        momentum or clear the weight derivatives).
        """
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            self.__dict__["d"+name].mult(factor)
    
    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)

    def setFactorHiddenMatrix(self, factToHid):
        assert(factToHid.shape == (self.numFact, self.numHid))
        assert(num.all(factToHid <= 0))
        self.factToHid = cm.CUDAMatrix(factToHid)
        self.factHidColNorm = num.sqrt(num.sum(factToHid**2)/factToHid.shape[1])
    
    def randomSparseFactToHid(self, connectionsPerHid = None):
        if connectionsPerHid == None:
            connectionsPerHid = self.numFact/20
        connectionsPerHid = max(1, connectionsPerHid)

        w = 1.0/connectionsPerHid

        factToHid = num.zeros((self.numFact, self.numHid))
        for i in range(connectionsPerHid):
            idx = num.random.randint(0, self.numFact, self.numHid)
            factToHid[idx, num.arange(self.numHid)] -= w

        self.setFactorHiddenMatrix(factToHid)
        self.factHidColNorm = num.sqrt(num.sum(factToHid**2)/factToHid.shape[1]) #we don't need this anymore
        
    def blockIdentityFactToHid(self, radius = 1):
        factToHid = -num.eye(self.numFact, self.numHid)
        for i in range(radius):
            factToHid -= num.eye(self.numFact, self.numHid, i+1)
            factToHid -= num.eye(self.numFact, self.numHid, -i-1)
        self.factToHidMask = cm.CUDAMatrix(factToHid)
        self.setFactorHiddenMatrix(factToHid)
        self.factHidColNorm = num.sqrt(num.sum(factToHid**2)/factToHid.shape[1]) #we don't need this anymore
    
    def constrainFactToHid(self):
        zeroOutPositives(self.factToHid, self.tempFactToHid)
        if self.factToHidMask != None:
            self.factToHid.mult(self.factToHidMask)

        #normalize columns in L1 sense
        self.factToHid.sum(axis=0, target = self.tempHidRow)
        self.tempHidRow.mult(-1)
        self.tempHidRow.reciprocal()
        self.factToHid.mult_by_row(self.tempHidRow) #unit L1 norm for columns
        
        #normalize columns in L2 sense
        #self.factToHid.mult(self.factToHid, target = self.tempFactToHid)
        #self.tempFactToHid.sum(axis = 0, target = self.tempHidRow)
        #cm.sqrt(self.tempHidRow)
        #self.tempHidRow.reciprocal()
        #self.factToHid.mult_by_row(self.tempHidRow)
        #self.factToHid.mult(self.factHidColNorm)

    def renormVisToFact(self):
        columnNorms(self.visToFact, self.tempVisToFact, self.curVisToFactColNorms)
        self.curVisToFactColNorms.reciprocal(target = self.tempFactRow)
        self.visToFact.mult_by_row(self.tempFactRow) #now columns of visToFact have unit norm
        
        self.curVisToFactColNorms.sum(axis=1, target = self.tempScalar)
        
        self.normVisToFact = 0.95*self.normVisToFact + (0.05/self.numFact)*self.tempScalar.asarray()[0,0]

        self.visToFact.mult(self.normVisToFact)
        
        #columnNorms(self.visToFact, self.tempVisToFact, self.curVisToFactColNorms)
        #
        #columnNorms(self.visToFact, self.tempVisToFact, self.tempFactRow)
        #self.tempFactRow.reciprocal()
        #self.visToFact.mult_by_row(self.tempFactRow)
        #
        #if self.maxColNorm == None:
        #    if not self.allColsSame:
        #        self.visToFact.mult_by_row(self.visToFactColNorms)
        #    else:
        #        #copies a 1 by 1 from gpu to cpu
        #        self.visToFact.mult( self.visToFactColNorms.sum(axis=1).asarray()[0,0]/self.numFact ) 
        #else:    
        #    #we constrain any columns with norm > self.maxColNorm to have norm == self.maxColNorm
        #    newNormsCPU = self.curVisToFactColNorms.asarray().copy()
        #    newNormsCPU[newNormsCPU > self.maxColNorm] = self.maxColNorm
        #    newNorms = cm.CUDAMatrix(newNormsCPU)
        #    self.visToFact.mult_by_row(newNorms)
            
            

        
    def step(self, data, renorm):
        if isinstance(data, cm.CUDAMatrix):
            self.vis = data
        else:
            self.vis = cm.CUDAMatrix(data)
        self.scaleDerivs(self.momentum)
        self.CD()
        self.decay()
        
        self.visToFact.add_mult(self.dvisToFact, self.learnRateVF/self.mbsz)
        self.factToHid.add_mult(self.dfactToHid, self.learnRateFH/self.mbsz)
        self.hidBias.add_mult(self.dhidBias, self.learnRateHB/self.mbsz)
        self.visBias.add_mult(self.dvisBias, self.learnRateVB/self.mbsz)

        self.constrainFactToHid()
        if renorm:
            self.renormVisToFact()
            
        
    def train(self, epochs, freshMinibatches):
        for ep in range(epochs):
            renorm = self.renormStartEpoch != None and ep > self.renormStartEpoch
            if self.renormStartEpoch != None and ep <= self.renormStartEpoch:
                columnNorms(self.visToFact, self.tempVisToFact, self.visToFactColNorms)
            if renorm:
                print "Constraining column norms of visToFact starting now!"
            for j,mb in enumerate(freshMinibatches()):
                self.step(mb, renorm)
                yield (ep, j)
    
    def hidActProbs(self, targ = None, vis = None):
        """
        targ had better be on the gpu or None
        """
        if targ == None:
            targ = self.hActProbs
        if vis == None:
            vis = self.vis
        
        #recall that self.acceleration calls self.hidActProbs
        normalizeInputData(vis, self.tempVisMB, self.sqColLens, self.tempRow, self.normalizedVisMB)
        
        #cm.dot(self.visToFact.T, vis, target = self.factResponses)
        cm.dot(self.visToFact.T, self.normalizedVisMB, target = self.factResponses)
        self.factResponses.mult(self.factResponses, target = self.factResponsesSq)
        cm.dot(self.factToHid.T, self.factResponsesSq, target = targ)
        
        targ.add_col_vec(self.hidBias)
        self.hNetInputs.assign(targ) #needed later in Hamiltonian computation
        targ.apply_sigmoid()

    def sampleHiddens(self, hActProbsOnGPU = None):
        if hActProbsOnGPU == None:
            hActProbsOnGPU = self.hActProbs
        self.hActs.fill_with_rand()
        self.hActs.less_than(hActProbsOnGPU, target = self.hActs)

    def CDStats(self, vis, normalizedVis, hid, posPhase):
        multiplier = 1.0 if posPhase else -1.0
        
        self.dhidBias.add_sums(hid, 1, mult = multiplier)
        self.dvisBias.add_sums(vis, 1, mult = multiplier)
        
        cm.dot(self.factToHid, hid, target = self.tempFactMB)
        self.tempFactMB.mult(self.factResponses)

        #I modified cudamat's add_dot to take a multiplier
        #need to multiply by 0.5 to make finite diffs agree
        self.dfactToHid.add_dot(self.factResponsesSq, hid.T, mult = 0.5*multiplier)
        if posPhase:
            #self.dvisToFact.add_dot(vis, self.tempFactMB.T)
            self.dvisToFact.add_dot(normalizedVis, self.tempFactMB.T)
        else:
            #self.dvisToFact.subtract_dot(vis, self.tempFactMB.T)
            self.dvisToFact.subtract_dot(normalizedVis, self.tempFactMB.T)

    def Hamiltonian(self, hamil):
        """
        This method computes the current value of the Hamiltonian for
        self.negVis and self.vel using the current weights.  We will
        produce a 1 by mbsz result and store it in hamil.

        This function depends on self.hNetInputs and self.sqColLens
        being set correctly.  So really this function depends on
        self.hidActProbs or self.acceleration (which calls
        self.hidActProbs) being called just before it is called.
        """
        #Potential energy
        #recall that self.acceleration calls self.hidActProbs
        logOnePlusExp(self.hNetInputs, self.tempHidMB, targ = self.tempHidMB)
        
        self.tempHidMB.sum(axis = 0, target = hamil)
        #vis bias contribution
        self.vis.mult_by_col(self.visBias, target = self.tempVisMB)
        hamil.add_sums(self.tempVisMB, axis=0)
        
        hamil.mult(-1)
        #quadratic visible term contribution, it is the opposite sign to the vis bias term
        hamil.add(self.sqColLens)
        
        #Kinetic energy
        self.vel.mult(self.vel, target = self.tempVisMB)
        hamil.add_sums(self.tempVisMB, axis = 0, mult = 0.5)

    def acceleration(self):
        #this sets self.hActProbs and self.normalizedVisMB and self.sqColLens
        self.hidActProbs(vis = self.negVis)
        
        cm.dot(self.factToHid, self.hActProbs, target = self.tempFactMB)
        self.tempFactMB.mult(-1)
        self.tempFactMB.mult(self.factResponses)
        #cm.dot(self.visToFact, self.tempFactMB, target = self.accel)
        cm.dot(self.visToFact, self.tempFactMB, target = self.normalizedAccel)

        #rename some things to be like Marc'Aurelio's code:
        normcoeff = self.tempRow2
        lengthsq = self.tempRow
        
        #these next few lines repeat some work, but it is too confusing to cache all this stuff at the moment
        self.sqColLens.mult(1.0/self.numVis, target = lengthsq)
        lengthsq.add(small) #self.tempRow is what Marc'Aurelio calls lengthsq
        cm.sqrt(lengthsq, target = normcoeff)
        normcoeff.mult(lengthsq) #now self.tempRow2 has what Marc'Aurelio calls normcoeff
        normcoeff.reciprocal()
        
        self.normalizedAccel.mult(self.negVis, target = self.tempVisMB)
        self.tempVisMB.sum(axis=0, target = self.tempRow3) #this tempRow stuff is getting absurd
        self.tempRow3.mult(-1.0/self.numVis)
        self.negVis.mult_by_row(self.tempRow3, target = self.tempVisMB)
        self.normalizedAccel.mult_by_row(lengthsq, target = self.accel)
        self.accel.add(self.tempVisMB)
        self.accel.mult_by_row(normcoeff)
        
        #quadratic in v term contribution to gradient
        self.accel.add(self.negVis)
        
        self.accel.mult(2) #all parts before this point have a 2 show up because of differentiation
        
        #vis bias contribution
        self.accel.add_col_mult(self.visBias, -1)
        
    def HMCSample(self, hActs = None):
        if hActs == None:
            hActs = self.hActs

        epsilon = self.hmcStepSize
        if self.stepSizeIsMean:
            epsilon = -self.hmcStepSize*num.log(1.0-num.random.rand())
        
        self.negVis.assign(self.vis)
        #sample a velocity and temporal direction
        self.vel.fill_with_randn()
        timeDir = 2*num.random.randint(2)-1
        
        self.Hamiltonian(self.prevHamil)
        
        #half-step
        self.acceleration() #updates self.accel
        self.vel.add_mult(self.accel, -0.5*timeDir*epsilon)
        self.negVis.add_mult(self.vel, timeDir*epsilon)
        #full leap-frog steps
        for s in range(self.hmcSteps-1):
            self.acceleration()
            self.vel.add_mult(self.accel, -timeDir*epsilon)
            self.negVis.add_mult(self.vel, timeDir*epsilon)
        #final half-step
        self.acceleration()
        self.vel.add_mult(self.accel, -0.5*timeDir*epsilon)
        self.negVis.add_mult(self.vel, timeDir*epsilon)
        
        self.Hamiltonian(self.hamil)
        
        #compute rejections
        self.prevHamil.subtract(self.hamil, target = self.thresh) #don't really need this new variable, but it is small
        cm.exp(self.thresh)
        self.tempRow.fill_with_rand()
        self.tempRow.less_than(self.thresh, target = self.tempRow) #tempRow entries are 0 for reject and 1 for accept
        self.tempRow.copy_to_host()
        rejRate = self.tempRow.numpy_array.sum()/float(self.mbsz)
        rejRate = 1-rejRate
        self.negVis.mult_by_row(self.tempRow) #zero out rejected columns
        negate(self.tempRow) #tempRow entries are 1 for reject and 0 for accept
        self.vis.mult_by_row(self.tempRow, target = self.tempVisMB)
        self.negVis.add(self.tempVisMB)

        smoothing = 0.9
        self.runningAvRej = smoothing*self.runningAvRej + (1.0-smoothing)*rejRate
        tol = 0.05
        #perhaps add this in later? right now the step size HAS to change unless it hits a max or min
        #if self.runningAvRej < self.targetRejRate*(1-tol) or self.runningAvRej < self.targetRejRate*(1+tol):
        #    pass
        if self.runningAvRej < self.targetRejRate:
            self.hmcStepSize = min(self.hmcStepSize*1.01, self.maxStepSize)
        else:
            self.hmcStepSize = max(self.hmcStepSize*0.99, self.minStepSize)
        
    def CD(self):
        """
        After this function runs we will have the negative data in
        self.negVis and self.hActProbs will hold the final hidden
        activation probabilities conditioned on the negative data.
        
        This function updates the weight derivative variables.
        """        
        #stores hidden activation probabilities in self.hActProbs
        self.hidActProbs()
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.normalizedVisMB, self.hActProbs, True)
        
        #updates self.hActs
        self.sampleHiddens(self.hActProbs)
        
        #updates self.negVis
        self.HMCSample()
        
        #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
        self.hidActProbs(vis = self.negVis)

        #compute negative phase statistics and subtract them from gradient variables
        self.CDStats(self.negVis, self.normalizedVisMB, self.hActProbs, False)

##    #for debugging
##    def dEdP(self, vis, hid, j,f):
##        vis.copy_to_host()
##        hid.copy_to_host()
##        v = vis.numpy_array.copy()
##        h = hid.numpy_array.copy()
##        self.visToFact.copy_to_host()
##        C = self.visToFact.numpy_array.copy()
        
##        fResp = 0.0
##        for i in range(self.numVis):
##            fResp += v[i,0]*C[i,f]
##        return -0.5*h[j,0]*fResp**2
    
##    def energy(self, vis, hid):
##        """
##        This function should only be used during debugging.
##        """
##        factResponses = cm.CUDAMatrix(reformat(num.zeros((self.numFact, self.mbsz))))
##        cm.dot(self.visToFact.T, vis, target = factResponses)
        
##        factHidTerm = cm.CUDAMatrix(reformat(num.zeros((self.numFact, self.mbsz))))
##        cm.dot(self.factToHid, hid, target = factHidTerm)
        
##        biasTerm = cm.CUDAMatrix(reformat(num.zeros((hid.shape))))
##        biasTerm.assign(hid)
##        biasTerm.mult_by_col(self.hidBias)
        
##        factResponses.mult(factResponses)
##        factResponses.mult(factHidTerm)
##        factResponses.mult(0.5)
        
##        row = cm.CUDAMatrix(reformat(num.zeros((1, self.mbsz))))
##        row.add_sums(factResponses, axis = 0)
##        row.add_sums(biasTerm, axis = 0)
        
##        energy1x1 = cm.CUDAMatrix(reformat(num.zeros((1,1))))
##        row.sum(axis=1, target = energy1x1)
        
##        energy1x1.copy_to_host()
##        energy = -1*energy1x1.numpy_array[0,0]
        
##        return energy
    
##    def energyCPU(self, vis, hid):
##        assert(vis.shape[1] == 1 == hid.shape[1])
##        vis.copy_to_host()
##        hid.copy_to_host()
##        v = vis.numpy_array.copy()
##        h = hid.numpy_array.copy()
##        self.visToFact.copy_to_host()
##        C = self.visToFact.numpy_array.copy()
##        self.factToHid.copy_to_host()
##        P = self.factToHid.numpy_array.copy()
##        self.hidBias.copy_to_host()
##        b = self.hidBias.numpy_array.copy()
        
##        term1 = 0.0
##        term2 = 0.0
##        for f in range(self.numFact):
##            fResp = 0.0
##            for i in range(self.numVis):
##                fResp += v[i,0]*C[i,f]
##            fRespSq = fResp**2
##            hTerm = 0.0
##            for j in range(self.numHid):
##                hTerm += h[j,0]*P[f,j]
##            term1 += hTerm*fRespSq
##        for j in range(self.numHid):
##            term2 += b[j,0]*h[j,0]
##        energy = -0.5*term1-term2
##        return energy

class MeanCovGRBM(CovGRBM):
    def __init__(self, numVis, numFact, numHid, numHidRBM, mbsz = 256, initWeightSigma = 0.02, initHidBiasRBM = 0):
        self.numHidRBM = numHidRBM
        self.visToHid = cm.CUDAMatrix(initWeightSigma*num.random.randn(numVis, self.numHidRBM))
        #self.visToHid = cm.CUDAMatrix(0.0*num.random.randn(numVis, self.numHidRBM))
        self.dvisToHid = cm.CUDAMatrix(num.zeros(self.visToHid.shape))
        self.hidBiasRBM = cm.CUDAMatrix(num.zeros((self.numHidRBM, 1)) + initHidBiasRBM)
        self.dhidBiasRBM = cm.CUDAMatrix(num.zeros(self.hidBiasRBM.shape))
        
        CovGRBM.__init__(self, numVis, numFact, numHid, mbsz, initWeightSigma)
        
    
    def initTemporary(self):
        CovGRBM.initTemporary(self)
        self.hActsRBM = cm.CUDAMatrix(num.zeros((self.numHidRBM, self.mbsz)))
        self.hActProbsRBM = cm.CUDAMatrix(num.zeros((self.numHidRBM, self.mbsz)))
        self.hNetInputsRBM = cm.CUDAMatrix(num.zeros((self.numHidRBM, self.mbsz)))

        self.tempRBMHidMB = cm.CUDAMatrix(num.zeros((self.numHidRBM, self.mbsz)))
        
    def weightVariableNames(self):
        """
        Returns the names of the variables for the weights that define
        this model in a cannonical order.
        """
        return "visToFact", "factToHid", "hidBias", "visBias", "visToHid", "hidBiasRBM"

    def printWeightNorms(self):
        learnRates = {"visToFact":self.learnRateVF, "factToHid":self.learnRateFH, \
                      "hidBias":self.learnRateHB, "visToHid":self.learnRateVH, \
                      "visBias":self.learnRateVB, "hidBiasRBM":self.learnRateHBRBM}
        d= dict((name, self.__dict__[name].euclid_norm()) for name in self.weightVariableNames())
        dd = dict(("d"+name, self.__dict__["d"+name].euclid_norm()/self.mbsz) for name in self.weightVariableNames())
        for name in self.weightVariableNames():
            print name+":", self.__dict__[name].euclid_norm(),",", \
                  learnRates[name]*self.__dict__["d"+name].euclid_norm()/self.mbsz, ";",
        #print d,dd

##    #@TODO: fix decay for mcRBM, add L1 decay to mean unit weights
##    def decay(self):
##        CovGRBM.decay()
##        #here are the learning parameters this method depends on
##        decayRate = self.weightCost
##        regType = self.regType
        
##        if decayRate > 0: #hopefully this saves time when decayRate == 0
##            #at the moment I don't feel like having L2 weight decay as an option
##            assert( regType in ["L1"] )
##            #assert( regType in ["L2","L1"] )
##            #if "L2" in regType:
##            #    self.visToFact.mult( 1-decayRate*self.learnRate )
##            #    #it doesn't really make sense to use L2 on factToHid since we keep the columns at a constant norm
##            if "L1" in regType:
##                self.updateSignOfWeights()
##                #self.visToFact.subtract_mult(self.signVisToFact, decayRate*self.learnRateVF)
##                self.dvisToFact.subtract_mult(self.signVisToFact, decayRate)
##                #self.factToHid.subtract_mult(self.signFactToHid, decayRate*self.learnRateFH)
##                self.dfactToHid.subtract_mult(self.signFactToHid, decayRate)
    
    def setLearningParams(self, **kwargs):
        CovGRBM.setLearningParams(self, **kwargs)
        
        self.learnRateVH = 0.02
        self.learnRateHBRBM = 0.004

        params = set(["learnRateVH", "learnRateHBRBM"])
        
        for k in params:
            if k in kwargs:
                self.__dict__[k] = kwargs[k]
    
    def sampleHiddensRBM(self, hActProbsOnGPU = None):
        if hActProbsOnGPU == None:
            hActProbsOnGPU = self.hActProbsRBM
        self.hActsRBM.fill_with_rand()
        self.hActsRBM.less_than(hActProbsOnGPU, target = self.hActsRBM)
    
    def hidActProbsRBM(self, vis = None):
        """
        targ had better be on the gpu or None
        """
        if vis == None:
            vis = self.vis
        targ = self.hActProbsRBM
        
        cm.dot( self.visToHid.T, vis, target = targ)
        targ.add_col_vec(self.hidBiasRBM)
        self.hNetInputsRBM.assign(targ) #needed later for Hamiltonian computation
        targ.apply_sigmoid()

    def CDStatsRBM(self, vis, hid, posPhase):
        """
        hid should be self.numHidRBM by mbsz and exist on the GPU
        vis should be self.numVis by mbsz and exist on the GPU

        We modify self.d$WEIGHT_NAME as a side effect.
        """
        multiplier = 1.0 if posPhase else -1.0
        self.dhidBiasRBM.add_sums(hid, 1, mult = multiplier)
        if posPhase:    
            self.dvisToHid.add_dot(vis, hid.T)
        else:
            self.dvisToHid.subtract_dot(vis, hid.T)
    
    def CDStats(self, vis, normalizedVis, hid, hidRBM, posPhase):
        CovGRBM.CDStats(self, vis, normalizedVis, hid, posPhase)
        self.CDStatsRBM(vis, hidRBM, posPhase)
    
    def CD(self):
        """
        After this function runs we will have the negative data in
        self.negVis and self.hActProbs will hold the final hidden
        activation probabilities conditioned on the negative data.
        
        This function updates the weight derivative variables.
        """        
        #stores hidden activation probabilities in self.hActProbs
        self.hidActProbs()
        #stores RBM hidden activation probabilities in self.hActProbsRBM
        self.hidActProbsRBM()
        
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.normalizedVisMB, self.hActProbs, self.hActProbsRBM, True)
        
        #updates self.hActs
        #self.sampleHiddens(self.hActProbs)
        #updates self.hActs
        #self.sampleHiddensRBM(self.hActProbsRBM)
        
        #updates self.negVis
        self.HMCSample()
        
        #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs and self.hActProbsRBM
        self.hidActProbs(vis = self.negVis)
        self.hidActProbsRBM(vis = self.negVis)
        
        #compute negative phase statistics and subtract them from gradient variables
        self.CDStats(self.negVis, self.normalizedVisMB, self.hActProbs, self.hActProbsRBM, False)

    def Hamiltonian(self, hamil):
        """
        This method computes the current value of the Hamiltonian for
        self.negVis and self.vel using the current weights.  We will
        produce a 1 by mbsz result and store it in hamil.

        This function depends on self.hNetInputs and self.hNetInputsRBM being set
        correctly.
        """
        CovGRBM.Hamiltonian(self, hamil) #kinetic term and CovGRBM potential term
        
        #all that remains is to add in the rbm potential term
        logOnePlusExp(self.hNetInputsRBM, self.tempRBMHidMB, targ = self.tempRBMHidMB)
        hamil.add_sums(self.tempRBMHidMB, axis=0, mult = -1.0)

    def acceleration(self):
        CovGRBM.acceleration(self)

        self.hidActProbsRBM(vis = self.negVis)
        self.accel.subtract_dot(self.visToHid, self.hActProbsRBM)
    
    def step(self, data, renorm):
        if isinstance(data, cm.CUDAMatrix):
            self.vis = data
        else:
            self.vis = cm.CUDAMatrix(data)
        self.scaleDerivs(self.momentum)
        self.CD()
        self.decay()

        self.visToFact.add_mult(self.dvisToFact, self.learnRateVF/self.mbsz)
        self.factToHid.add_mult(self.dfactToHid, self.learnRateFH/self.mbsz)
        self.hidBias.add_mult(self.dhidBias, self.learnRateHB/self.mbsz)
        self.visBias.add_mult(self.dvisBias, self.learnRateVB/self.mbsz)
        self.visToHid.add_mult(self.dvisToHid, self.learnRateVH/self.mbsz)
        self.hidBiasRBM.add_mult(self.dhidBiasRBM, self.learnRateHBRBM/self.mbsz)
        
        self.constrainFactToHid()
        if renorm:
            self.renormVisToFact()
        
    def train(self, epochs, freshMinibatches):
        for ep in range(epochs):
            renorm = self.renormStartEpoch != None and ep > self.renormStartEpoch
            columnNorms(self.visToFact, self.tempVisToFact, self.curVisToFactColNorms)
            if self.renormStartEpoch != None and ep <= self.renormStartEpoch:
                columnNorms(self.visToFact, self.tempVisToFact, self.visToFactColNorms)
            if renorm:
                print "Constraining column norms of visToFact starting now!"
            for j,mb in enumerate(freshMinibatches()):
                self.step(mb, renorm)
                yield (ep, j)

    def fprop(self, minibatch, negatePrecUnits = True):
        assert(self.mbsz == minibatch.shape[1])
        if isinstance(minibatch, cm.CUDAMatrix):
            self.vis = minibatch
        else:
            self.vis = cm.CUDAMatrix(minibatch)

        self.hidActProbs()
        if negatePrecUnits:
            negate(self.hActProbs)
        self.hidActProbsRBM()

        outputDims = self.numHid + self.numHidRBM
        output = cm.CUDAMatrix(num.zeros((outputDims, self.mbsz)))
        
        output.set_row_slice(0, self.numHid, self.hActProbs)
        output.set_row_slice(self.numHid, outputDims, self.hActProbsRBM)
        return output
        
    
    def features(self, inp, negatePrecUnits = True):
        mbsz = self.mbsz
        numcases = inp.shape[1]
        numFullMinibatches = numcases / mbsz
        excess = numcases % mbsz
        feat = []
        
        for i in range(numFullMinibatches):
            idx = i*mbsz
            acts = self.fprop(inp[:,idx:idx+mbsz], negatePrecUnits)

            feat.append(acts.asarray().copy())
            
        if excess != 0:
            idx = numFullMinibatches*mbsz
            mb = num.zeros((inp.shape[0], mbsz))
            mb[:,:excess] = inp[:, idx:]
            acts = self.fprop(mb)

            feat.append(acts.asarray()[:,:excess].copy())
        return num.hstack(feat)
    
def mcRBMPreprocessor(net, negatePrecUnits = True):
    def prepro(minibatch):
        return net.fprop(minibatch, negatePrecUnits)
    return prepro


def main():
    batch_size = 128
    # load data
    d = loadmat('patches_16x16x3.mat') # input in the format PxD (P vectorized samples with D dimensions)
    totnumcases = d["dataraw"].shape[0]
    numBatches = totnumcases/batch_size
    d = d["dataraw"][0:int(totnumcases/batch_size)*batch_size,:].copy() 
    totnumcases = d.shape[0]
    # preprocess input
    dd = loadmat("pca_projections.mat")
    d = num.dot(dd["transform"],d.T).copy() # get the PCA projections
    data = cm.CUDAMatrix(reformat(d))
    
    net = CovGRBM(d.shape[0], 400, 400, mbsz = 128, initWeightSigma = 0.02)
    
    d = loadmat("topo2D_3x3_stride1_400filt.mat")
    net.setFactorHiddenMatrix(-d["w2"])

    net.hmcSteps = 20


    freshData = lambda : (data.slice(b*batch_size, (b+1)*batch_size) for b in range(numBatches))
    highestEp = -1
    for ep, mb in net.train(100, freshData, 10, True):
        if ep > highestEp:
            highestEp = ep
            print "Epoch %d" % (highestEp)
            print net.runningAvRej, net.hmcStepSize
            print "V2F:", net.visToFact.euclid_norm()
        
    #for ep in range(100):
    #    print "Epoch %d" % (ep)
    #    print net.runningAvRej, net.hmcStepSize
    #    print "V2F:", net.visToFact.euclid_norm()
    #    
    #    for b in range(numBatches):
    #        mb = data.slice(b*batch_size, (b+1)*batch_size)
    #        net.step(mb)
            
    
    
def test():
    num.random.seed(10)
    m = 1
    net = CovGRBM(8, 4, 16, mbsz = m, initWeightSigma = 0.5)
    
    vis = cm.CUDAMatrix(reformat(2*num.random.randn(8,m)))
    #hid = cm.CUDAMatrix(reformat(2*num.random.rand(16,16)))

    net.vis = vis
    net.hidActProbs()
    delta = 0.00001

    print net.energy(vis, net.hActProbs)
    
    
    
    #get derivs we compute with update rules
    net.CDStats(net.vis, net.hActProbs, True)
    net.dvisToFact.copy_to_host()
    dvisToFact = net.dvisToFact.numpy_array.copy()
    net.dfactToHid.copy_to_host()
    dfactToHid = net.dfactToHid.numpy_array.copy()
    net.dhidBias.copy_to_host()
    dhidBias = net.dhidBias.numpy_array.copy()
    
    print net.energyCPU(vis, net.hActProbs)
    
    net.visToFact.copy_to_host()
    vToF = net.visToFact.numpy_array.copy()
    vToFA = vToF.copy()
    vToFB = vToF.copy()
    vToFB[2,3] += delta
    vToFA[2,3] -= delta
    net.visToFact = cm.CUDAMatrix(reformat(vToFB))
    EB = net.energy(vis, net.hActProbs)
    net.visToFact = cm.CUDAMatrix(reformat(vToFA))
    EA = net.energy(vis, net.hActProbs)
    deriv = (EB-EA)/(2*delta)
    print "deriv:", dvisToFact[2,3]
    print "finite differences:", deriv
    net.visToFact = cm.CUDAMatrix(reformat(vToF))

    print
    print net.dEdP(vis, net.hActProbs, 3,2)
    fToH = net.factToHid.numpy_array.copy()
    fToHA = fToH.copy()
    fToHB = fToH.copy()
    fToHB[2,3] += delta
    fToHA[2,3] -= delta
    net.factToHid = cm.CUDAMatrix(reformat(fToHB))
    EB = net.energy(vis, net.hActProbs)
    net.factToHid = cm.CUDAMatrix(reformat(fToHA))
    EA = net.energy(vis, net.hActProbs)
    deriv = (EB-EA)/(2*delta)
    print "deriv:", dfactToHid[2,3]
    print "finite differences:", deriv
    net.factToHid = cm.CUDAMatrix(reformat(fToH))

    bias = net.hidBias.numpy_array.copy()
    biasA = bias.copy()
    biasB = bias.copy()
    biasB[2,0] += delta
    biasA[2,0] -= delta
    net.hidBias = cm.CUDAMatrix(reformat(biasB))
    EB = net.energy(vis, net.hActProbs)
    net.hidBias = cm.CUDAMatrix(reformat(biasA))
    EA = net.energy(vis, net.hActProbs)
    deriv = (EB-EA)/(2*delta)
    print "deriv:", dhidBias[2,0]
    print "finite differences:", deriv
    net.hidBias = cm.CUDAMatrix(reformat(bias))
    
                        
import gpu_lock

if __name__ == "__main__":
    print "export LD_LIBRARY_PATH=/u/gdahl/cudaLearn/"
    print "export CUDAMATDIR=/u/gdahl/cudaLearn"
    
    devId = gpu_lock.obtain_lock_id()
    cm.cuda_set_device(devId)
    
    cm.cublas_init()
    cm.CUDAMatrix.init_random(1)
    #test()
    main()
    cm.cublas_shutdown()

#Copyright (c) 2009,2010 George Dahl

from rbm_gpu import *
import ctypes as ct
from singleSoftmax import singleSoftmax

#from mcGRBM_gpu import MCGRBM

_libebm = ct.cdll.LoadLibrary("libebm.so")

class GaussianBinaryDBNLayer(GaussianRBM):
    def __init__(self, numVis, numHid, mbsz = 256, initHidBias = 0.0, initWeightSigma = 0.05):
        #superclass __init__ should call our initTemporary and thus create the inputError variable
        GaussianRBM.__init__(self, numVis, numHid, mbsz, initHidBias, initWeightSigma)
        #will be a view of the subsequent layer's input error
        self.errSignal = None
        
    def initTemporary(self):
        GaussianRBM.initTemporary(self)
        #the "return value" of bprop
        self.inputError = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))

    def fprop(self):
        self.hidActProbs()
        return self.hActProbs
    
    def bprop(self):
        """
        Before calling this function you MUST set self.errSignal and
        self.vis correctly.  Also, fprop should have been called to
        make sure self.hActProbs is set up (which would have required
        self.vis anyway).
        """
        #self.errSignal should be numHid by mbsz
        _libebm.mult_by_sigmoid_deriv(self.hActProbs.p_mat, self.errSignal.p_mat)
        
        cm.dot(self.visToHid, self.errSignal, target = self.inputError)

        #apply momentum
        self.scaleDerivs(self.momentum)
        self.dhidBias.add_sums(self.errSignal, 1)
        self.dvisToHid.add_dot(self.vis, self.errSignal.T)

        return self.inputError

    def applySupervisedUpdates(self):
        """
        This function should ONLY be called right after bprop has been
        called.
        """
        assert(self.regType in ["L2","L1"])
        self.decay()
        self.hidBias.add_mult(self.dhidBias, self.learnRate/self.mbsz)
        self.visToHid.add_mult(self.dvisToHid, self.learnRate/self.mbsz)
        
class BinaryBinaryDBNLayer(GaussianBinaryDBNLayer):
    #be sure to set the learning rate
    def visActProbs(self):
        GaussianBinaryDBNLayer.visActProbs(self)
        self.negVis.apply_sigmoid()

def softmax(netInput):
    #we expect netInput to be numClasses by mbsz
    result = netInput - num.max(netInput, 0).reshape(1,netInput.shape[1])
    result = num.exp(result)
    result /= num.sum(result, 0).reshape(1,netInput.shape[1])
    return result

def sigmoid(netInput):
    return 1/(1+num.exp(-netInput))

class KClassCrossEntropy(object):
    def __init__(self, numIn, numClasses, mbsz = 256):
        self.numIn = numIn
        self.numClasses = numClasses
        self._mbsz = mbsz

        #self.classSpecificCosts = None
        self.W = cm.CUDAMatrix(0.05*num.random.randn(numIn, numClasses))
        self.dW = cm.CUDAMatrix(num.zeros((numIn, numClasses)))
        self.bias = cm.CUDAMatrix(num.zeros((numClasses, 1)))
        self.dbias = cm.CUDAMatrix(num.zeros((numClasses, 1)))
        
        self.inpt = None
        self.errSignal = None
        
        self.initTemporary()
        self.setLearningParams(0.001, 0.9, 0, "L2")

    @property
    def numVis(self):
        return self.numIn

    def getVis(self): return self.inpt
    def setVis(self, newInpt): self.inpt = newInpt
    vis = property(getVis, setVis)
    
    def initTemporary(self):
        self.actsGPU = cm.CUDAMatrix(num.zeros((self.numClasses, self.mbsz)))
        self.inputError = cm.CUDAMatrix(num.zeros((self.numIn, self.mbsz)))

    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)

    def packWeights(self):
        d = {}
        self.W.copy_to_host()
        d["W"] = self.W.numpy_array.copy()
        self.bias.copy_to_host()
        d["bias"] = self.bias.numpy_array.copy()
        return d
        
    def loadWeights(self, wDict):
        for w_name in wDict:
            assert( self.__dict__.has_key(w_name) )
            w = wDict[w_name]
            assert( self.__dict__[w_name].shape == w.shape )
            self.__dict__[w_name] = cm.CUDAMatrix(w)
    
    def setLearningParams(self, learnRate, momentum, weightCost, regType):
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightCost = weightCost
        self.regType = regType
    
    def fprop(self):
        cm.dot( self.W.T, self.inpt, target = self.actsGPU)
        self.actsGPU.add_col_vec(self.bias)

        self.actsGPU.copy_to_host()
        self.acts = softmax(self.actsGPU.numpy_array)
        return self.acts

    def outputErr(self, target):
        assert(target.shape == self.acts.shape)
        self.errSignal = cm.CUDAMatrix(target - self.acts)
        return self.errSignal.numpy_array
    
    def bprop(self):
        """
        This method assumes self.inpt holds the activations of the
        previous layer and resides on the GPU.  It also assume that
        self.outputErr has just been called.
        """
        cm.dot(self.W, self.errSignal, target = self.inputError)

        #apply momentum
        self.dW.mult(self.momentum)
        self.dbias.mult(self.momentum)
        
        self.dbias.add_sums(self.errSignal, 1)
        self.dW.add_dot(self.inpt, self.errSignal.T)
        
        return self.inputError
    
    def applySupervisedUpdates(self):
        """
        This function should ONLY be called right after bprop has been
        called.
        """
        self.decay()
        self.bias.add_mult(self.dbias, self.learnRate/self.mbsz)
        self.W.add_mult(self.dW, self.learnRate/self.mbsz)
    
    def decay(self):
        if self.weightCost > 0: #and self.classSpecificCosts == None:
            assert(self.regType in ["L2"]) #we only support L2 right now, should add L1 later
            self.W.mult( 1-self.learnRate*self.weightCost/self.mbsz )
        #if self.classSpecificCosts != None:
        #    pass
    
    def error(self, targets):
        """
        This function returns the value of the error function
        outputErr is returning the derivative of.  This function
        depends on self.acts being set correctly with fprop so please
        do not call it unless you know that is the case.
        
        This particular error function is the k class cross entropy
        error (in bits).

        We perform this computation entirely on the CPU.
        """
        #self.acts is numclasses by numcases
        #targets is numclasses by numcases
        #err = targets*num.log(self.acts)/num.log(2)
        tiny = 0.000000001
        err = targets*num.log(self.acts+tiny)/num.log(2)
        if num.any(num.isnan(err)):
            raise RuntimeError("0 log 0 produced NaN in cross entropy computation")
        if num.any(num.isneginf(err)):
            raise RuntimeError("log 0 produces -Inf in cross entropy computation")
        #we may want to divide by the number of cases to get avg err per case
        return -num.sum(err) #will sum over both dimensions


class BinaryOutputsCrossEntropy(KClassCrossEntropy):
    def __init__(self, numIn, numOut, mbsz = 256):
        self.numIn = numIn
        self.numOut = numOut
        self._mbsz = mbsz
        
        self.W = cm.CUDAMatrix(0.05*num.random.randn(numIn, numOut))
        self.dW = cm.CUDAMatrix(num.zeros((numIn, numOut)))
        self.bias = cm.CUDAMatrix(num.zeros((numOut, 1)))
        self.dbias = cm.CUDAMatrix(num.zeros((numOut, 1)))
        
        self.inpt = None
        self.errSignal = None
        
        self.initTemporary()
        self.setLearningParams(0.08, 0.9, 0, "L2")
    
    def initTemporary(self):
        self.acts = cm.CUDAMatrix(num.zeros((self.numOut, self.mbsz)))
        self.errSignal = cm.CUDAMatrix(num.zeros((self.numOut, self.mbsz)))
        self.inputError = cm.CUDAMatrix(num.zeros((self.numIn, self.mbsz)))
    
    def fprop(self):
        cm.dot( self.W.T, self.inpt, target = self.acts)
        self.acts.add_col_vec(self.bias)
        self.acts.apply_sigmoid()
        return self.acts

    def outputErr(self, targetsOnCPU):
        targetsOnGPU = cm.CUDAMatrix(targetsOnCPU)
        targetsOnGPU.subtract(self.acts, target = self.errSignal)
                
    def error(self, targets):
        """
        This function returns the value of the error function
        outputErr is returning the derivative of.  This function
        depends on self.acts being set correctly with fprop so please
        do not call it unless you know that that is the case.
        
        This particular error function is the binary cross entropy
        error (in bits) summed over the different output units.

        We perform this computation entirely on the CPU.
        """
        #self.acts is numclasses by numcases
        #targets is numclasses by numcases
        #err = targets*num.log(self.acts)/num.log(2)
        tiny = 0.000000001
        err = targets*num.log(self.acts+tiny)/num.log(2)
        if num.any(num.isnan(err)):
            raise RuntimeError("0 log 0 produced NaN in cross entropy computation")
        if num.any(num.isneginf(err)):
            raise RuntimeError("log 0 produces -Inf in cross entropy computation")
        #we may want to divide by the number of cases to get avg err per case
        return -num.sum(err) #will sum over both dimensions


#in the middle of writing this, need to upgrade to latest cudamat
class GeneralizedSoftmax(KClassCrossEntropy):
    def __init__(self, numIn, codeMatrix, mbsz):
        """
        Code matrix should be a numClasses by d matrix that holds the
        binary code vector for each class.
        """
        self.numIn = numIn
        self.numOut = codeMatrix.shape[1]
        self.numClasses = codeMatrix.shape[0]
        self._mbsz = mbsz

        self.C = cm.CUDAMatrix(codeMatrix) #numClasses by numOut
        self.W = cm.CUDAMatrix(0.05*num.random.randn(numIn, self.numOut))
        self.dW = cm.CUDAMatrix(num.zeros((numIn, self.numOut)))
        self.bias = cm.CUDAMatrix(num.zeros((self.numOut, 1)))
        self.dbias = cm.CUDAMatrix(num.zeros((self.numOut, 1)))
        
        self.inpt = None
        self.errSignal = None
        
        self.initTemporary()
        self.setLearningParams(0.08, 0.9, 0, "L2")
    
    def initTemporary(self):
        self.outputsGPU = cm.CUDAMatrix(num.zeros((self.numClasses, self.mbsz)))
        self.acts = cm.CUDAMatrix(num.zeros((self.numOut, self.mbsz)))
        self.errSignal = cm.CUDAMatrix(num.zeros((self.numOut, self.mbsz)))
        self.inputError = cm.CUDAMatrix(num.zeros((self.numIn, self.mbsz)))
    
    def fprop(self):
        cm.dot( self.W.T, self.inpt, target = self.acts)
        self.acts.add_col_vec(self.bias)
        cm.dot(self.C, self.acts, target = self.outputsGPU)
        
        self.outputsGPU.copy_to_host()

        self.outputs = softmax(self.outputsGPU.numpy_array)
        
        return self.outputs

    def outputErr(self, target):
        assert(target.shape == self.outputs.shape)
        self.outputErrGPU = cm.CUDAMatrix(target - self.outputs)
        cm.dot(self.C.T, self.outputErrGPU, target = self.errSignal)

    def error(self, targets):
        tiny = 0.000000001
        err = targets*num.log(self.outputs+tiny)/num.log(2)
        if num.any(num.isnan(err)):
            raise RuntimeError("0 log 0 produced NaN in cross entropy computation")
        if num.any(num.isneginf(err)):
            raise RuntimeError("log 0 produces -Inf in cross entropy computation")
        #we may want to divide by the number of cases to get avg err per case
        return -num.sum(err) #will sum over both dimensions

class HybridKClassCrossEntropy(object):
    """
    This layer can't (conveniently) be pretrained because it needs
    labels to even do CD!  So we only support fine tuning of the
    layer, which does involve a generative component.
    """
    def __init__(self, numVis, numHid, numClasses, mbsz = 256):
        self.numVis = numVis
        self.numHid = numHid
        self.numClasses = numClasses
        self._mbsz = mbsz

        
        self.W = cm.CUDAMatrix(0.05*num.random.randn(numVis, numHid))
        self.dW = cm.CUDAMatrix(num.zeros((numVis, numHid)))
        self.U = cm.CUDAMatrix(0.05*num.random.randn(numClasses, numHid))
        self.dU = cm.CUDAMatrix(num.zeros((numClasses, numHid)))

        self.labBias = cm.CUDAMatrix(num.zeros((numClasses, 1)))
        self.hidBias = cm.CUDAMatrix(num.zeros((numHid, 1)))
        self.visBias = cm.CUDAMatrix(num.zeros((numVis, 1)))
        
        self.dLabBias = cm.CUDAMatrix(num.zeros((numClasses, 1)))
        self.dHidBias = cm.CUDAMatrix(num.zeros((numHid, 1)))
        self.dVisBias = cm.CUDAMatrix(num.zeros((numVis, 1)))
        
        
        self.vis = None
        self.labels = None
        self.errSignal = None
        
        self.initTemporary()
        self.setLearningParams(0.08, 0.9, 0, "L2", 1, False, 0)

    def initTemporary(self):
        self.tempW = cm.CUDAMatrix(num.zeros((self.numVis, self.numHid)))
        self.tempU = cm.CUDAMatrix(num.zeros((self.numClasses, self.numHid)))
        
        self.inputError = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.hActs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.hActProbs = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.negVis = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.negLabels = cm.CUDAMatrix(num.zeros((self.numClasses, self.mbsz)))
        self.tempVisMB = cm.CUDAMatrix(num.zeros((self.numVis, self.mbsz)))
        self.tempHidMB = cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz)))
        self.tempLabelMB = cm.CUDAMatrix(num.zeros((self.numClasses, self.mbsz)))
        self.tempRow = cm.CUDAMatrix(num.zeros((1, self.mbsz)))
        self.tempLabelCol = cm.CUDAMatrix(num.zeros((self.numClasses, 1)))
        self.tempHidCol = cm.CUDAMatrix(num.zeros((self.numHid, 1)))
        self.tempHidRow = cm.CUDAMatrix(num.zeros((1, self.numHid)))
        self.disc_grad_labhid = cm.CUDAMatrix(num.zeros((self.numClasses, self.numHid)))
        self.discacchids = []
        for c in range(self.numClasses):
            self.discacchids.append(cm.CUDAMatrix(num.zeros((self.numHid, self.mbsz))))
        self.negHidActProbs = None
    
    def setLearningParams(self, learnRate, momentum, weightCost, regType, cdSteps, PCD, genCost):
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightCost = weightCost
        self.regType = regType
        self.cdSteps = cdSteps
        self.doPCD = PCD
        #the factor we multiply the generative weight updates by (in addition to the gloabl learning rate)
        self.genCost = genCost
        
    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)

    def fprop(self):
        cm.dot( self.W.T, self.vis, target = self.hActProbs)
        self.hActProbs.add_col_vec(self.hidBias)
        
        self.U.copy_to_host()
        labHid = self.U.numpy_array
        Utrans = cm.CUDAMatrix(labHid.transpose())
                
        self.tempLabelMB.assign_scalar(0.0)
        self.tempLabelMB.add_col_vec(self.labBias)
        
        for c in range(self.numClasses): #183 of these
            self.discacchids[c].assign_scalar(0.0) #numHid by mbsz
            self.discacchids[c].add(self.hActProbs)
            self.discacchids[c].add_col_vec(Utrans.slice(c,c+1))
            self.tempHidMB.assign(self.discacchids[c])
            cm.exp(self.tempHidMB)
            self.tempHidMB.add_scalar(1.0)
            cm.log(self.tempHidMB)
            self.tempHidMB.sum(0, self.tempRow)
            self.tempLabelMB.set_row_slice(c,c+1, self.tempRow)
        #singleSoftmax(self.tempLabelMB, self.tempLabelCol, self.tempRow)
        singleSoftmax(self.tempLabelMB, self.tempRow)
        self.tempLabelMB.copy_to_host()
        self.acts = self.tempLabelMB.numpy_array.copy() #self.tempLabelMB.numpy_array #we may not need the copy here
        return self.acts
    
    def outputErr(self, target):
        assert(target.shape == self.acts.shape)
        self.errSignal = cm.CUDAMatrix(target - self.acts)
        self.labels = cm.CUDAMatrix(target)
        return self.errSignal.numpy_array

    def applyMomentum(self):
        self.dW.mult(self.momentum)
        self.dU.mult(self.momentum)
        self.dLabBias.mult(self.momentum)
        self.dHidBias.mult(self.momentum)
        self.dVisBias.mult(self.momentum)
    
    def bprop(self):
        """
        We assume fprop has just been called and that self.labels has
        been set properly.
        """
        self.applyMomentum()
        
        self.tempHidMB.assign_scalar(0.0)
        for c in range(self.numClasses):
            self.discacchids[c].apply_sigmoid()
            self.errSignal.get_row_slice(c, c+1, self.tempRow)
            self.discacchids[c].mult_by_row(self.tempRow)
            self.discacchids[c].sum(1, self.tempHidCol)
            self.tempHidCol.reshape((1,self.numHid))
            self.disc_grad_labhid.set_row_slice(c,c+1, self.tempHidCol)
            self.tempHidCol.reshape((self.numHid,1))
            self.tempHidMB.add(self.discacchids[c])
        cm.dot(self.W, self.tempHidMB, target = self.inputError)
        self.dLabBias.add_sums(self.errSignal, 1)
        self.dU.add(self.disc_grad_labhid)
        self.dHidBias.add_sums(self.tempHidMB, 1)
        self.dW.add_dot(self.vis, self.tempHidMB.T)

        #should call CD here for the generative part of the updates
        if self.genCost > 0:
            self.CDn(clampLabels = True)
        
        return self.inputError
    
    def applySupervisedUpdates(self):
        """
        This function should ONLY be called right after bprop has been
        called.
        """
        self.decay()
        self.W.add_mult(self.dW, self.learnRate/self.mbsz)
        self.U.add_mult(self.dU, self.learnRate/self.mbsz)
        self.labBias.add_mult(self.dLabBias, self.learnRate/self.mbsz)
        self.hidBias.add_mult(self.dHidBias, self.learnRate/self.mbsz)
        self.visBias.add_mult(self.dVisBias, self.learnRate/self.mbsz)
        #print "W:", self.W.euclid_norm()
        #print "U:", self.U.euclid_norm()
    
    def decay(self):
        if self.weightCost > 0:
            assert(self.regType in ["L2"]) #we only support L2 right now, should add L1 later
            self.W.mult( 1-self.learnRate*self.weightCost/self.mbsz )
            self.U.mult( 1-self.learnRate*self.weightCost/self.mbsz )
        
    def step(self, dataOnGPU, labelsOnGPU):
        """
        This function performs one step of supervised generative 'pretraining'.
        """
        self.vis = dataOnGPU
        self.labels = labelsOnGPU
        self.applyMomentum()
        self.CDn(False)
        self.applySupervisedUpdates()
    
    def packWeights(self):
        d = {}
        self.W.copy_to_host()
        d["W"] = self.W.numpy_array.copy()
        self.U.copy_to_host()
        d["U"] = self.U.numpy_array.copy()
        self.labBias.copy_to_host()
        d["labBias"] = self.labBias.numpy_array.copy()
        self.hidBias.copy_to_host()
        d["hidBias"] = self.hidBias.numpy_array.copy()
        self.visBias.copy_to_host()
        d["visBias"] = self.visBias.numpy_array.copy()
        return d
        
    def loadWeights(self, wDict):
        for w_name in wDict:
            assert( self.__dict__.has_key(w_name) )
            w = wDict[w_name]
            assert( self.__dict__[w_name].numpy_array.shape == w.shape )
            self.__dict__[w_name] = cm.CUDAMatrix(w)

    
                
    def error(self, targets):
        """
        This function returns the value of the error function
        outputErr is returning the derivative of.  This function
        depends on self.acts being set correctly with fprop so please
        do not call it unless you know that is the case.
        
        This particular error function is the k class cross entropy
        error (in bits).

        We perform this computation entirely on the CPU.
        """
        #self.acts is numclasses by numcases
        #targets is numclasses by numcases
        #err = targets*num.log(self.acts)/num.log(2)
        tiny = 0.00000001
        err = targets*num.log(self.acts+tiny)/num.log(2)
        if num.any(num.isnan(err)):
            raise RuntimeError("0 log 0 produced NaN in cross entropy computation")
        if num.any(num.isneginf(err)):
            raise RuntimeError("log 0 produces -Inf in cross entropy computation")
        #we may want to divide by the number of cases to get avg err per case
        return -num.sum(err) #will sum over both dimensions
    
    def curRecErr(self):
        self.vis.subtract(self.negVis, target = self.tempVisMB)
        self.labels.subtract(self.negLabels, target = self.tempLabelMB)
        vErr = self.tempVisMB.euclid_norm()
        lErr = self.tempLabelMB.euclid_norm()
        #print vErr, lErr
        return (vErr+lErr)**2
        #return vErr**2
    
    def sampleHiddens(self, hActProbsOnGPU = None):
        if hActProbsOnGPU == None:
            hActProbsOnGPU = self.hActProbs
        self.hActs.fill_with_rand()
        self.hActs.less_than(hActProbsOnGPU, target = self.hActs)
    
    def hidActProbs(self, targ = None, vis = None, labels = None):
        """
        targ had better be on the gpu or None
        """
        if targ == None:
            targ = self.hActProbs
        if vis == None:
            vis = self.vis
        if labels == None:
            labels = self.labels
        
        cm.dot( self.W.T, vis, target = targ)
        targ.add_dot(self.U.T, labels)
        targ.add_col_vec(self.hidBias)
        targ.apply_sigmoid()

    
    def visActProbs(self, reconstructLabels):
        #since we always have labels clamped, we have no need for a negLabels variable
        cm.dot( self.W, self.hActs, target = self.negVis)
        self.negVis.add_col_vec(self.visBias)
        self.negVis.apply_sigmoid()
        if reconstructLabels:
            cm.dot( self.U, self.hActs, target = self.negLabels)
            self.negLabels.add_col_vec(self.labBias)
            #singleSoftmax(self.negLabels, self.tempLabelCol, self.tempRow)
            singleSoftmax(self.negLabels, self.tempRow)

    
    def CDStats(self, vis, labels,  hid, posPhase, clampLabels):
        """
        hid should be self.numHid by mbsz and exist on the GPU
        vis should be self.numVis by mbsz and exist on the GPU

        We modify self.d$WEIGHT_NAME as a side effect.
        """
        multiplier = 1.0 if posPhase else -1.0
        multiplier *= self.genCost
        
        self.dHidBias.add_sums(hid, 1, mult = multiplier)
        self.dVisBias.add_sums(vis, 1, mult = multiplier)

        if not clampLabels:
            self.dLabBias.add_sums(labels, 1, mult = multiplier)
        
        if posPhase:    
            #self.dW.add_dot(vis, hid.T)
            cm.dot(vis, hid.T, target = self.tempW)
            self.dW.add_mult(self.tempW, self.genCost)
            #self.dU.add_dot(labels, hid.T)
            cm.dot(labels, hid.T, target = self.tempU)
            self.dU.add_mult(self.tempU, self.genCost)
        else:
            #self.dW.subtract_dot(vis, hid.T)
            cm.dot(vis, hid.T, target = self.tempW)
            self.dW.subtract_mult(self.tempW, self.genCost)
            #self.dU.subtract_dot(labels, hid.T)
            cm.dot(labels, hid.T, target = self.tempU)
            self.dU.subtract_mult(self.tempU, self.genCost)
    
    def CDn(self, clampLabels):
        """
        After this function runs we will have the negative data in
        self.negVis and self.hActProbs will hold the final hidden
        activation probabilities conditioned on the negative data.
        
        This function updates the weight derivative variables.
        """
        #we depend on the following two learning parameters
        n = self.cdSteps
        
        #stores hidden activation probabilities in self.hActProbs
        self.hidActProbs()
        #compute positive phase statistics and add them to gradient variables
        self.CDStats(self.vis, self.labels, self.hActProbs, True, clampLabels)
        
        for i in range(n):
            #updates self.hActs
            self.sampleHiddens(self.hActProbs)

            if clampLabels:
                #updates self.negVis
                self.visActProbs(reconstructLabels = False)
            else:
                #updates self.negVis and self.negLabels
                self.visActProbs(reconstructLabels = True)
            #stores recomputed (based on self.negVis) hidden act probs in self.hActProbs
            self.hidActProbs(vis = self.negVis)

        if clampLabels:
            #compute negative phase statistics and subtract them from gradient variables
            self.CDStats(self.negVis, self.labels, self.hActProbs, False, clampLabels)
        else:
            #compute negative phase statistics and subtract them from gradient variables
            self.CDStats(self.negVis, self.negLabels, self.hActProbs, False, clampLabels)
            
     #should add PCD back in and test self.doPCD flag


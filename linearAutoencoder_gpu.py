#Copyright (c) 2009,2010 George Dahl

import numpy as num
import cudamat as cm
from cudamat import reformat

class LinearAutoencoder(object):
    def __init__(self, numVis, numHid, mbsz = 256, initWeightSigma = 0.01):
        self._mbsz = mbsz
        self.numVis, self.numHid = numVis, numHid
        self.visToHid = initWeightSigma*num.random.randn(numVis, numHid)
        self.hidToVis = self.visToHid.transpose().copy()#initWeightSigma*num.random.randn(numHid, numVis)#

        self.init_weight_storage()
        self.initTemporary()
        self.inp = None

        self.learnRate = 0.0001
        self.momentum = 0.9
        

    def getMBSZ(self):
        return self._mbsz
    
    def setMBSZ(self, newMBSZ):
        self._mbsz = newMBSZ
        self.initTemporary()
    mbsz = property(getMBSZ,setMBSZ)

    def packWeights(self):
        d = {}
        self.visToHid.copy_to_host()
        d["visToHid"] = self.visToHid.numpy_array.copy()
        self.hidToVis.copy_to_host()
        d["hidToVis"] = self.hidToVis.numpy_array.copy()
        return d

    def loadWeights(self, wDict):
        for w_name in self.weightVariableNames():
            assert( wDict.has_key(w_name) )
            w = wDict[w_name]
            assert( self.__dict__[w_name].numpy_array.shape == wDict[w_name].shape )
            self.__dict__[w_name] = cm.CUDAMatrix(reformat(w))
        
    
    def weightVariableNames(self):
        return "visToHid", "hidToVis"
    
    def init_weight_storage(self):
        for name in self.weightVariableNames():
            w = self.__dict__[name]
            self.__dict__[name] = cm.CUDAMatrix(reformat(w))
            self.__dict__["d"+name] = cm.CUDAMatrix(reformat(0.0 * w))

    
    def initTemporary(self):
        self.hid = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.out = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))
        self.delta = cm.CUDAMatrix(reformat(num.zeros((self.numHid, self.mbsz))))
        self.tempVisMB = cm.CUDAMatrix(reformat(num.zeros((self.numVis, self.mbsz))))

    def encode(self):
        cm.dot(self.visToHid.T, self.inp, target = self.hid)

    def decode(self):
        cm.dot(self.hidToVis.T, self.hid, target = self.out)
    
    def fprop(self):
        """
        self.inp must reside on the gpu and be initialized correctly.
        """
        self.encode()
        self.decode()
    
    def curRecErr(self):
        self.inp.subtract(self.out, target = self.tempVisMB)
        return self.tempVisMB.euclid_norm()**2
    
    def bprop(self):
        #apply momentum
        self.dhidToVis.scalar_mult(self.momentum)
        self.dvisToHid.scalar_mult(self.momentum)
        
        #NB: we have the opposite sign convention here from the usual way
        self.out.subtract(self.inp) # compute error
        
        self.dhidToVis.add_dot(self.hid, self.out.T)
        
        cm.dot(self.hidToVis, self.out, target = self.delta)

        self.dvisToHid.add_dot(self.inp, self.delta.T)

    def step(self, data):
        if isinstance(data, cm.CUDAMatrix):
            self.inp = data
        else:
            self.inp = cm.CUDAMatrix(reformat(data))
        self.fprop()
        recErr = self.curRecErr()
        self.bprop()
        for j, wname in enumerate(self.weightVariableNames()):
            #NOTE THE UNUSUAL SIGN CONVENTION HERE
            self.__dict__[wname].subtract_mult( self.__dict__["d"+wname], self.learnRate/self.mbsz ) 
        return recErr

    def train(self, freshData, epochs):
        for ep in range(epochs):
            recErr = 0.0
            numMinibatches = 0
            for mb in freshData():
                recErr += self.step(mb)
                numMinibatches += 1
            yield recErr/numMinibatches/self.numVis/self.mbsz

    

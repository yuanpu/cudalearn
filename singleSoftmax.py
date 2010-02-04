#Copyright (c) 2009,2010 George Dahl

import numpy as num

import cudamat as cm
from cudamat import reformat

import time

def cpuSoftmax(netInput):
    #we expect netInput to be numClasses by mbsz
    result = netInput - num.max(netInput, 0).reshape(1,netInput.shape[1])
    result = num.exp(result)
    result /= num.sum(result, 0).reshape(1,netInput.shape[1])
    return result


def singleSoftmax(netInputs, tempCol, tempRow):
    """
    We modify netInputs in place to hold the softmax activation
    probabilities and compute them in a numerically stable way.
    """
    assert(tempCol.shape[0] == netInputs.shape[0])
    assert(tempRow.shape[1] == netInputs.shape[1])
    assert(tempRow.shape[0] == tempCol.shape[1])

    tempCol.assign_scalar(1)
    netInputs.max(axis = 0, target = tempRow)
    netInputs.subtract_dot(tempCol, tempRow)
    cm.exp(netInputs)
    netInputs.sum(axis = 0, target = tempRow)
    tempRow.reciprocal()
    netInputs.mult_by_row(tempRow)

def maskedSingleSoftmax(netInputs, tempMatrix, sMask, notSMask, onesCol, tempRow):
    """
    We now assume we have a single k way softmax and some number of
    Gaussian units.  So we only want to apply the softmax activation
    function to the first k rows of netInputs.
    """
    assert(onesCol.shape[0] == netInputs.shape[0])
    assert(tempRow.shape[1] == netInputs.shape[1])
    assert(tempRow.shape[0] == onesCol.shape[1])
    assert(netInputs.shape == tempMatrix.shape == sMask.shape == notSMask.shape)
    
    c = num.finfo(num.float32).min/16
    assert(num.exp(c+200) == 0.0)
    netInputs.mult(sMask, target = tempMatrix)
    tempMatrix.add_mult(notSMask, c)
    tempMatrix.max(axis = 0, target = tempRow)
    #onesCol.assign_scalar(1)
    tempMatrix.subtract_dot(onesCol, tempRow)
    cm.exp(tempMatrix)
    tempMatrix.sum(axis = 0, target = tempRow)
    tempRow.reciprocal()
    tempMatrix.mult_by_row(tempRow)
    netInputs.mult(notSMask)
    tempMatrix.mult(sMask)
    netInputs.add(tempMatrix)

def testMaskedSM():
    x = num.random.randn(16+21,1024)
    k = 16
    x = x.astype("float32")
    r = x.copy()
    t = time.time()
    xGPU = cm.CUDAMatrix(reformat(x))
    r[:k,:] = cpuSoftmax(r[:k,:])
    print time.time()-t
    
    tempCol = cm.CUDAMatrix(reformat(num.ones((xGPU.shape[0],1))))
    tempRow = cm.CUDAMatrix(reformat(num.zeros((1,xGPU.shape[1]))))
    tempMatrix = cm.CUDAMatrix(reformat(num.zeros(xGPU.shape)))
    sMask = num.zeros(xGPU.shape)
    sMask[:k,:] = 1
    notSMask = 1-sMask
    sMask = cm.CUDAMatrix(reformat(sMask))
    notSMask = cm.CUDAMatrix(reformat(notSMask))

    t = time.time()
    maskedSingleSoftmax(xGPU, tempMatrix, sMask, notSMask, tempCol, tempRow)
    print time.time()-t
    
    xGPU.copy_to_host()
    
    
    
    diff = r-xGPU.numpy_array
    #print diff
    print num.sum(num.abs(diff))
    
    
def main():
    x = num.random.randn(5,64)
    x = x.astype("float32")
    xGPU = cm.CUDAMatrix(reformat(x))
    r = cpuSoftmax(x)
    print r.dtype
    tempCol = cm.CUDAMatrix(reformat(num.zeros((xGPU.shape[0],1))))
    tempRow = cm.CUDAMatrix(reformat(num.zeros((1,xGPU.shape[1]))))
    singleSoftmax(xGPU, tempCol, tempRow)
    xGPU.copy_to_host()
    diff = xGPU.numpy_array-r
    print num.sum(num.abs(diff))
    testMaskedSM()

    col = cm.CUDAMatrix(reformat(num.random.rand(5,1)))
    print col.shape
    col.copy_to_host()
    print col.numpy_array
    col.reshape((1,5))
    print col.shape
    col.copy_to_host()
    print col.numpy_array
    garb = cm.CUDAMatrix(reformat(num.zeros((5,5))))
    garb.set_row_slice(2,3,col)
    garb.copy_to_host()
    print garb.numpy_array
    
if __name__ == "__main__":
    print "export LD_LIBRARY_PATH=/u/gdahl/cudaLearn/"
    print "export CUDAMATDIR=/u/gdahl/cudaLearn"
    
    devId = cm.cuda_get_free_device()
    cm.cuda_set_device(devId)
    
    cm.cublas_init()
    cm.CUDAMatrix.init_random(1)
    main()
    cm.cublas_shutdown()

from __future__ import print_function
import sys
sys.setrecursionlimit(15000)
from six.moves import cPickle
import itertools
import numpy as np
from random import shuffle
import theano
from theano import tensor as T
import lasagne
from lasagne.nonlinearities import leaky_rectify, softmax, linear, sigmoid
from lasagne.init import HeNormal
from lasagne.layers import InputLayer, Conv3DLayer, MaxPool3DLayer, DenseLayer, batch_norm, ConcatLayer, Upscale3DLayer, get_output_shape, dropout

def createUNet(X, filterNumStart, depth, unet_lp):
    unetwork = lasagne.layers.InputLayer(shape=(None, 1, unet_lp["tile"], unet_lp["tile"], unet_lp["tile"]), input_var = X)
    connections = []
    HE = HeNormal(gain='relu')
    for d in range(depth):
        unetwork = Conv3DLayer(unetwork, filterNumStart*(2**d), (3,3,3), W=HE,
                              nonlinearity=leaky_rectify)
        if unet_lp["dropout"]!=0:
            unetwork = dropout(unetwork, p=unet_lp["dropout"])
        unetwork = batch_norm(unetwork)
        print("---", get_output_shape(unetwork))
        unetwork = Conv3DLayer(unetwork, filterNumStart*(2**(d*2)), (3,3,3), W=HE,
                              nonlinearity=leaky_rectify)
        unetwork = batch_norm(unetwork)
        print ("---", get_output_shape(unetwork))
        if d!=depth-1:
            connections.append(unetwork)
            unetwork = MaxPool3DLayer(unetwork, pool_size=(2,2,2))
            print ("------------------------downto", get_output_shape(unetwork))

    if unet_lp["dropout"]!=0:
        unetwork = dropout(unetwork, p=unet_lp["dropout"])

    for d in range(depth-1):

        unetwork = batch_norm(Upscale3DLayer(unetwork, 2))
        print ("-------------------upto", get_output_shape(unetwork))
        print ("concating with ", get_output_shape(connections[-1-d]))
        unetwork = ConcatLayer([connections[-1-d], unetwork], cropping = [None, None, 'center', 'center', 'center'])
        print ("---", get_output_shape(unetwork))

        unetwork = Conv3DLayer(unetwork, filterNumStart*(2**(depth-1-d)), (3,3,3), W=HE,
                              nonlinearity=leaky_rectify)
        if unet_lp["dropout"]!=0:
            unetwork = dropout(unetwork, p=unet_lp["dropout"])
        unetwork = batch_norm(unetwork)
        print ("---", get_output_shape(unetwork))

        unetwork = Conv3DLayer(unetwork, filterNumStart*(2**(depth-1-d)), (3,3,3), W=HE,
                              nonlinearity=leaky_rectify)
        unetwork = batch_norm(unetwork)
        print ("---", get_output_shape(unetwork))

    unetwork = Conv3DLayer(unetwork, 1, (1,1,1), W=HE, nonlinearity=sigmoid)
    sh = get_output_shape(unetwork)
    print ("---", sh)
    return [unetwork, sh[2], (unet_lp["tile"]-sh[2])//2]

def createClassificationNet(X, filterNumStart, cnet_lp):
    print("create network")
    network = lasagne.layers.InputLayer(shape=(None, 1, cnet_lp["tile"], cnet_lp["tile"], cnet_lp["tile"]), input_var = X)
    HE = HeNormal(gain='relu')

    # CONV LAYER 1
    network = Conv3DLayer(network, filterNumStart*2, (5,5,5), W=HE,
                  nonlinearity=leaky_rectify)
    if cnet_lp["dropout"]!=0:
        network = dropout(network, p=cnet_lp["dropout"])
    network = batch_norm(network)

    # MAXPOOL 1
    network = MaxPool3DLayer(network, pool_size=(2,2,2))

    # CONV LAYER 2
    network = Conv3DLayer(network, filterNumStart*4, (3,3,3), W=HE,
                  nonlinearity=leaky_rectify)
    if cnet_lp["dropout"]!=0:
        network = dropout(network, p=cnet_lp["dropout"])
    network = batch_norm(network)

    print("Convolutional layers DONE")

    # MAXPOOL 2
    network = MaxPool3DLayer(network, pool_size=(2,2,2))

    # DENSE LAYER 1
    network = DenseLayer(network, num_units=32,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    print("Dense 1")
    # DENSE LAYER 2
    network = DenseLayer(network, num_units=1,
        nonlinearity=lasagne.nonlinearities.sigmoid)
        #nonlinearity=lasagne.nonlinearities.softmax)

    print("return network")
    return network

def loadSegmentFunction(X, unetwork):
    print ("        creating U network output...")
    predictedSegmentation = lasagne.layers.get_output(unetwork)
    pred_function = theano.function([X], predictedSegmentation, allow_input_downcast=True, on_unused_input='ignore')
    return pred_function

def loadClassifierFunction(X, cnetwork):
    print ("        creating clasifier network output...")
    predictedClass = lasagne.layers.get_output(cnetwork)
    pred_function = theano.function([X], predictedClass, allow_input_downcast=True, on_unused_input='ignore')
    return pred_function

# START
netsPath = "model"

ftensor5 = T.TensorType('float32', (False,)*5)
X = ftensor5()

unet_lp={"tile":148,"dropout":0.25}
cnet_lp={"tile":148,"dropout":0.25}

print ("creating networks...")
[unetwork, outdim, margin] = createUNet(X, 32, 4, unet_lp)
cnetwork = createClassificationNet(X, 32, cnet_lp)

print ("creating functions...")
classify_fn = loadClassifierFunction(X, cnetwork)
segment_fn = loadSegmentFunction(X, unetwork)

with open('cnet.save', 'wb') as f:
    cPickle.dump(cnetwork, f, protocol=cPickle.HIGHEST_PROTOCOL)
with open('unet.save', 'wb') as f:
    cPickle.dump(unetwork, f, protocol=cPickle.HIGHEST_PROTOCOL)
with open('cfunc.save', 'wb') as f:
    cPickle.dump(classify_fn, f, protocol=cPickle.HIGHEST_PROTOCOL)
with open('ufunc.save', 'wb') as f:
    cPickle.dump(segment_fn, f, protocol=cPickle.HIGHEST_PROTOCOL)


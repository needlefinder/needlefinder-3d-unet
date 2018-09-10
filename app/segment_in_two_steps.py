from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot
import sys
import itertools
import numpy as np
from random import shuffle
import SimpleITK as sitk
from scipy import ndimage
from sklearn.cluster import DBSCAN
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

def load_net_parameters(network, fname):
    with np.load(fname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
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

def segmentCase(classify_fn, segment_fn, case, lp, outdim, margin):
    print ("segmenting....")

    tile_with_margin=outdim+2*margin
    featuresBatch = np.zeros((1, 1, tile_with_margin, tile_with_margin, tile_with_margin), dtype='float32')

    mri_sitk = sitk.ReadImage(case)
    mri = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
    sh = mri.shape
    quantile1 = np.percentile(mri.flatten(), 1)
    quantile99 = np.percentile(mri.flatten(), 99)
    mri=np.clip(mri, quantile1, quantile99)
    mri -= mri.min()
    mri /= mri.max()
    pad_size = int((lp["tile"] - outdim)/2.0)
    mri = np.pad(mri, pad_size, "symmetric")

    out = np.zeros(sh, dtype="float32")

    coords = list(itertools.product(range(0,sh[0], outdim), range(0,sh[1], outdim), range(0,sh[2], outdim)))
    print ("||", coords)

    shuffle(coords)
    xlim = sh[0]-outdim
    ylim = sh[1]-outdim
    zlim = sh[2]-outdim
    for coord in coords:
        xx = min(coord[0],xlim)
        yy = min(coord[1],ylim)
        zz = min(coord[2],zlim)

        featuresBatch[0,0,:,:,:] = mri[xx:xx+tile_with_margin,yy:yy+tile_with_margin,zz:zz+tile_with_margin]

        predicted_class = float(classify_fn(featuresBatch)[0][0])
        print("--------------------", predicted_class)
        if predicted_class <= 0.5:
            tile_to_segment = False
        elif predicted_class > 0.5:
            tile_to_segment = True

        if tile_to_segment:
            print("TILE WITH NEEDLES!")
            res = segment_fn(featuresBatch)
            out[xx:xx+outdim, yy:yy+outdim, zz:zz+outdim] = res[0,0,:,:,:]
        else:
            print("BLACK TILE!")
            out[xx:xx+outdim, yy:yy+outdim, zz:zz+outdim] = 0

    out = out.astype(np.float32)
    out -= out.min()
    out /= out.max()
    out[out>0.5]=1.0
    out[out<=0.5]=0.0

    out_sitk = sitk.GetImageFromArray(out.astype(np.uint8))
    out_sitk.CopyInformation(mri_sitk)
    out_sitk = sitk.Cast(out_sitk, sitk.sitkUInt8)

    return out_sitk

def STAPLE_voting(imgs, cw=1.0):
    voting = sitk.STAPLEImageFilter()
    voting.SetConfidenceWeight(cw)
    voted_sitk = voting.Execute(imgs)
    voted_np = sitk.GetArrayFromImage(voted_sitk).astype(np.float32)

    voted_np -= voted_np.min()
    voted_np /= voted_np.max()
    voted_np[voted_np>0.5]=1.0
    voted_np[voted_np<=0.5]=0.0

    #bin_voted_sitk = sitk.Cast(sitk.GetImageFromArray(voted_np), sitk.sitkUInt8)
    #bin_voted_sitk.CopyInformation(voted_sitk)

    return voted_np.astype(np.uint8)

def execute_DBscan(img, threshold=0.9, dbscan_eps=12):
    nonzero_coords=np.array(np.nonzero(img)).T
    db = DBSCAN(eps=dbscan_eps, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(nonzero_coords)
    labels = db.labels_
    # noise has label equal to -1

    # compute optimal number of clusters
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    no_cl = np.zeros_like(img, dtype=int)
    h=matplotlib.pyplot.hist(y_db, bins=n_clusters_)
    lbl_max=np.argmax(h[0])-1
    for i,v in enumerate(nonzero_coords):
        if(y_db[i]==lbl_max):
            no_cl[v[0],v[1],v[2]]=1

    return no_cl.astype(np.float32)

def remove_small_island(img, intensity_threshold=0.8, island_threshold=50):

    img[img<=intensity_threshold]=0
    img[img>intensity_threshold]=1
    img=img.astype(np.uint8)

    label_im, nb_labels = ndimage.label(img)
    sizes = ndimage.sum(img, label_im, range(nb_labels + 1))
    mask_size = sizes < island_threshold
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    label_im[label_im!=0]=1
    return label_im.astype(np.uint8)

def WriteImage(img_np, output_fn, ref_img_sitk):
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkUInt8)
    img_sitk.CopyInformation(ref_img_sitk)
    sitk.WriteImage(img_sitk, output_fn)

def segment_and_vote(case, out_fn):

    netsPath = "model"

    ftensor5 = T.TensorType('float32', (False,)*5)
    X = ftensor5()

    training_epochs={}
    training_epochs["0"]={"classifier":12, "unet":26}
    training_epochs["1"]={"classifier":4, "unet":41}
    training_epochs["2"]={"classifier":36, "unet":24}
    training_epochs["3"]={"classifier":56, "unet":45}
    training_epochs["4"]={"classifier":13, "unet":39}

    unet_lp={"tile":148,"dropout":0.25}
    cnet_lp={"tile":148,"dropout":0.25}

    print ("creating networks...")
    [unetwork, outdim, margin] = createUNet(X, 32, 4, unet_lp)
    cnetwork = createClassificationNet(X, 32, cnet_lp)

    print ("creating functions...")
    classify_fn = loadClassifierFunction(X, cnetwork)
    segment_fn = loadSegmentFunction(X, unetwork)
    unetwork = load_net_parameters(unetwork, "%s/2/%d.npz" % (netsPath, training_epochs["2"]["unet"]))
    cnetwork = load_net_parameters(cnetwork, "%s/2/CLASSIFIER_%d.npz" % (netsPath, training_epochs["2"]["classifier"]))

    segmentation_sitk = segmentCase(classify_fn, segment_fn, case, unet_lp, outdim, margin)
    sitk.WriteImage(segmentation_sitk, out_fn)

    """
    CODE FOR MULTISEGMENTATION STRATEGY
    segmentations=[]
    for i in range(5):

        print(i)
        unetwork = load_net_parameters(unetwork, "%s/%s/%d.npz" % (netsPath, i, training_epochs[str(i)]["unet"]))
        cnetwork = load_net_parameters(cnetwork, "%s/%s/CLASSIFIER_%d.npz" % (netsPath, i, training_epochs[str(i)]["classifier"]))

        segmentations.append(segmentCase(classify_fn, segment_fn, case, unet_lp, outdim, margin))

    staple_voted = STAPLE_voting(segmentations, 1.0)
    WriteImage(staple_voted, out_fn, segmentations[0])
    #dbscan_cleaned = execute_DBscan(staple_voted, threshold=0.9, dbscan_eps=7)
    #WriteImage(dbscan_cleaned, "DBoutput.nrrd", segmentations[0])
    #removed_island_img = remove_small_island(dbscan_cleaned, intensity_threshold=0.8, island_threshold=50)
    #WriteImage(removed_island_img, out_fn, segmentations[0])
    """


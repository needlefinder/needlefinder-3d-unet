from fns.utils import *

def plot_prediction(x_test, y_test, prediction, save=False):

    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12, 12), sharey=True, sharex=True)

    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)

    ax = np.atleast_3d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i, 0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i, 1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i, 2])
        if i == 0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, nz, channels].
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    offset2 = (data.shape[3] - shape[3]) // 2
    out = data[:, offset0:(-offset0), offset1:(-offset1), offset2:(-offset2)]
    # out = data[:, offset0:offset0 + shape[1], offset1:offset1 + shape[2], offset2:offset2 + shape[3]]
    return out


def reshape_to_shape(data, shape, padding):
    """
    Crops the array to the given image shape by removing the border (expects shape [nx, ny, nz])
    :param data: the array to crop
    :param shape: the target shape
    """

    Tempshape = (319, 319, 255)  ## the max shape of original data
    if padding == "noise":
        std = np.std(data)
        mean = np.mean(data)
        temp = np.random.normal(std, mean, Tempshape)
    elif padding == "zero":
        temp = np.zeros(Tempshape, dtype=np.bool)
    elif padding == "ones":
        temp = np.ones(Tempshape, dtype=np.bool)
    else:
        raise ValueError("padding must be noise or zero")

    offset0 = (temp.shape[0] - data.shape[0]) // 2
    offset1 = (temp.shape[1] - data.shape[1]) // 2
    offset2 = (temp.shape[2] - data.shape[2]) // 2
    temp[offset0:offset0 + data.shape[0], offset1:offset1 + data.shape[1], offset2:offset2 + data.shape[2]] = data[:, :,
                                                                                                              :]

    offset0_ = (temp.shape[0] - shape[0]) // 2
    offset1_ = (temp.shape[1] - shape[1]) // 2
    offset2_ = (temp.shape[2] - shape[2]) // 2
    out = temp[offset0_:offset0_ + shape[0], offset1_:offset1_ + shape[1], offset2_:offset2_ + shape[2]]

    #     print(out.shape)
    return out


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_devonc(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W, keep_prob_):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_3d, keep_prob_)


def deconv3d(x, W, stride=1):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='VALID')


def max_pool(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
    return tf.div(exponential_map, evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[4]]))
    return tf.div(exponential_map, tensor_sum_exp)


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")
    #   return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 5d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, 0, idx), (1, -1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    img_z = tf.shape(img)[3]
    V = tf.reshape(V, tf.stack((img_w, img_h, img_z, 1)))
    V = tf.transpose(V, (3, 0, 1, 2))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, img_z, 1)))
    return V


def loadCases(p):
    f = open(p)
    res = []
    for l in f:
        l = l[:-1]
        if l == "":
            break
        if l[-1] == '\r':
            l = l[:-1]
        res.append(l)
    return res


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))

def cutVolume(data, tile_in=60, tile=148):
    '''
    Cut the volume in smaller volumes, overlaping so the FOV of the unet (60x60x60) is cover in every location of the 
    original volume, padded on the boundaries
    '''
    
    ### pad volume
    print("Original input shape", data.shape)
    data = np.pad(data,((44,44),(44,44), (44,44)), mode='mean')

    Mx, My, Mz = data.shape
    kx = Mx//tile_in + 1*((Mx%tile_in)>0)
    ky = Mx//tile_in + 1*((My%tile_in)>0)
    kz = Mz//tile_in + 1*((Mz%tile_in)>0)
    print('Padded input shape:', data.shape)
    print('# of parts', kx,ky,kz)

    off_x = 60
    off_y = 60
    off_z = 60

    arr_data = []
    nbTiles = 0
    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                # to not go over the boundaries
                x = min(off_x*i, Mx - tile)
                y = min(off_y*j, My - tile)
                z = min(off_z*k, Mz - tile)
                x = np.int(x)
                y = np.int(y)
                z = np.int(z)
                # print(x,y,z)
                data_s = data[x : x + tile, y : y + tile, z : z + tile ]
                arr_data.append(data_s)
                nbTiles += 1
                # stop cutting if next part is over the boundaries
                if (off_z*(k+1)) > (Mz - tile):
                    break
            if (off_y*(j+1)) > (My - tile):
                    break
        if (off_x*(i+1)) > (Mx - tile):
                    break
    print("number of tiles: %d " % nbTiles)
    arr_data = np.array(arr_data)
    return arr_data

def predict_full_volume(net, arr_data, model_path="./unet_trained/model 6.cpkt"):
    '''
    Perform inference on subvolumes
    '''
    arr_out = []
    for i in trange(arr_data.shape[0]):
        img = arr_data[i]
        img = img[np.newaxis,...,np.newaxis]
        #input shape size required 1,148,148,148,1
        img -= np.amin(img)
        img /= np.amax(img)
        out = net.predict(model_path, img)[0][:,:,:,0]
        # out = np.ones((60,60,60))*i
        out_p = np.pad(out,((44,44),(44,44), (44,44)), mode='constant', constant_values=[0])
        arr_out.append(out_p)
    return arr_out

def recombine(arr_out, data, tile_in=60, tile=148):
    '''
    Recombine subvolume into original shape
    '''
    data = np.pad(data,((44,44),(44,44), (44,44)), mode='constant', constant_values=[0])
    Mx, My, Mz = data.shape
    kx = Mx//tile_in + 1*((Mx%tile_in)>0)
    ky = Mx//tile_in + 1*((My%tile_in)>0)
    kz = Mz//tile_in + 1*((Mz%tile_in)>0)
    off_x = 60
    off_y = 60
    off_z = 60
    data = np.zeros((Mx, My, Mz))
    l=-1   
    print('-'*50)
    print('Padded input shape:', data.shape)
    print('# of parts', kx,ky,kz)

    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                l+=1
                x = min(off_x*i, Mx - tile)
                y = min(off_y*j, My - tile)
                z = min(off_z*k, Mz - tile)
                x = np.int(x)
                y = np.int(y)
                z = np.int(z)
                data[x : x + tile, y : y + tile, z : z + tile ] += arr_out[l]
                if (off_z*(k+1)) > (Mz - tile):
                    break
            if (off_y*(j+1)) > (My - tile):
                    break
        if (off_x*(i+1)) > (Mx - tile):
                    break

    print("# of subvolumes merged: ", l+1)
    data = np.array(data)
    # data[np.where(data<l//2)]=0
    # data[np.where(data>=l//2)]=1
    data = data.astype(np.int8)
    data=data[44:-44,44:-44,44:-44]
    print(np.unique(data, return_counts=True))
    print(data.shape)
    return data
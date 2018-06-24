from fns.utils import *
from fns.functions import *

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """

    channels = 1
    n_class = 1

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        ### minimum number of label voxel set to 1
        self.thresh = 2500
        self.check_vol = False
        
    def _count_valid_training(self, datafile, debug=False):
        c=0
        vals = []
        for i in trange(len(datafile)):
            label = self._next_data_label(datafile)
            sum_label = np.sum(label[44:-44,44:-44,44:-44])
            vals.append(sum_label)
            if sum_label > self.thresh:
                c +=1
        return c, vals

    def _load_data_and_label(self, datafile, batch_size=1):
        if batch_size == 1:
            data, label = self._next_data(datafile)

            data = self._process_data(data)
            label = self._process_labels(label)

            data, label = self._post_process(data, label)

            nx = data.shape[0]
            ny = data.shape[1]
            nz = data.shape[2]

            return data.reshape(1, nx, ny, nz, self.channels), label.reshape(1, nx, ny, nz, self.n_class)
        else:
            data_ = []
            label_ = []
            
            for i in range(batch_size):
                data, label = self._next_data(datafile)
                #if self.check_vol:
                 #   sum_label = np.sum(label[44:-44,44:-44,44:-44])
                  #  if sum_label < self.thresh:
                   #     while sum_label < self.thresh:
                    #        data, label = self._next_data(datafile)
                #else:
                 #   data, label = self._next_data(datafile)

                data = self._process_data(data)
                label = self._process_labels(label)

                data, label = self._post_process(data, label)

                data_.append(data)
                label_.append(label)

            nx = data.shape[0]
            ny = data.shape[1]
            nz = data.shape[2]

            data_ = np.array(data_)
            label_ = np.array(label_)

            return data_.reshape(batch_size, nx, ny, nz, self.channels), label_.reshape(batch_size, nx, ny, nz, self.n_class)


    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[0]
            ny = label.shape[1]
            nz = label.shape[2]
            labels = np.zeros((nx, ny, nz, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = 1 - label
            return labels

        return label

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        return data, labels

    def _post_process_clahe(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        nx,ny,nz = data.shape
        clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(30, 30))
        data_res = []
        for k in range(nz):
            cl = data[..., k].copy()
            cl = cl.astype('uint16')
            data_res.append(clahe.apply(cl))
        data_res = np.swapaxes(data_res, 0, 1)
        data_res = np.swapaxes(data_res, 1, 2)
        return data_res, labels

    def __call__(self, array=None, datafile="training", batch_size=1):
        if array != None:
            if datafile == "training":
                self.trainingfile_idx += 1
                if self.trainingfile_idx >= len(array):
                    self.trainingfile_idx = 0
                data = self._process_data(array[self.trainingfile_idx][0])
                label = self._process_labels(array[self.trainingfile_idx][1])
            elif datafile == "validation":
                self.validationfile_idx += 1
                if self.validationfile_idx >= len(array):
                    self.validationfile_idx = 0
                data = self._process_data(array[self.validationfile_idx][0])
                label = self._process_labels(array[self.validationfile_idx][1])
            elif datafile == "testing":
                self.testingfile_idx += 1
                if self.testingfile_idx >= len(array):
                    self.testingfile_idx = 0
                data = self._process_data(array[self.testingfile_idx][0])
                label = self._process_labels(array[self.testingfile_idx][1])
            else:
                raise NameError("No such datafile, datafile must be training, validation or testing")

            data, label = self._post_process(data, label)
            nx = data.shape[0]
            ny = data.shape[1]
            nz = data.shape[2]
            return data.reshape(1, nx, ny, nz, self.channels), label.reshape(1, nx, ny, nz, self.n_class)

        else:
            if datafile == "training":
                data, label = self._load_data_and_label(self.training_data_files, batch_size=batch_size)
            elif datafile == "validation":
                data, label = self._load_data_and_label(self.validation_data_files, batch_size=batch_size)
            elif datafile == "testing":
                data, label = self._load_data_and_label(self.testing_data_files, batch_size=batch_size)
            else:
                raise NameError("No such datafile, datafile must be training, validation or testing")
            return data, label


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix e.g. 'train/fish_1.tif'
    and 'train/fish_1_mask.tif'
    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'

    """

    def __init__(self, array=False, search_path='', a_min=None, a_max=None, data_suffix=".tif", 
                 split_vol=False, check_vol=False,mask_suffix='_mask.tif'):

        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.trainingfile_idx = -1
        self.validationfile_idx = -1
        self.testingfile_idx = -1
        self.split_vol = split_vol
        self.check_vol = check_vol

        if array == True:
            pass

        else:
            self.training_data_files, self.validation_data_files, self.testing_data_files = self._find_data_files()
            assert len(self.training_data_files) > 0, "No training files"
            assert len(self.validation_data_files) > 0, "No validation files"
            assert len(self.testing_data_files) > 0, "No testing files"

            print("Number of training data used: %s" % len(self.training_data_files))
            print("Number of validation data used: %s" % len(self.validation_data_files))
            print("Number of testing data used: %s" % len(self.testing_data_files))

        self.channels = 1

    def _find_data_files(self):
        #         rootPath = "/home/administrator/GynNeedleFinder/preprocessed_data/"
        rootPath = "/home/gp1514/DATA/"
        dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
        claheDataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"

        trainingCases = loadCases("training.txt")
        validationCases = loadCases("validation.txt")
        testingCases = loadCases("testing.txt")
        
        files_training_ = [claheDataPath + name + '/case.nrrd' for name in trainingCases]
        files_validation_ = [claheDataPath + name + '/case.nrrd' for name in validationCases]
        files_testing_ = [claheDataPath + name + '/case.nrrd' for name in testingCases]
        
        if self.split_vol:
            print(50*'-')
            print("Using split volumes")
            files_training, files_validation, files_testing = [], [], []

            #for file in files_training_: 
            #    folder = file.replace('case.nrrd', '')
            #    files_training.append(glob.glob(folder+'case_*.nrrd'))
            #for file in files_validation_: 
            #    folder = file.replace('case.nrrd', '')
            #    files_validation.append(glob.glob(folder+'case_*.nrrd'))
            #for file in files_testing_: 
            #    folder = file.replace('case.nrrd', '')
            #    files_testing.append(glob.glob(folder+'case_*.nrrd'))

            #files_training = list(np.concatenate(files_training))
            #files_validation = list(np.concatenate(files_validation))
            #files_testing = list(np.concatenate(files_testing))
            # print(files_training_)
            # print(50*'*')
            # print(files_training)
            with open('training_subvolumes.txt', 'r') as f:
                files_training = f.read().splitlines()
            with open('validation_subvolumes.txt', 'r') as f:
                files_validation = f.read().splitlines()
            with open('testing_subvolumes.txt', 'r') as f:
                files_testing = f.read().splitlines()
                
                
                
        else:
            files_training, files_validation, files_testing = files_training_, files_validation_, files_testing_
            
        np.random.shuffle(files_training)
        np.random.shuffle(files_validation)
        np.random.shuffle(files_testing)
        return files_training, files_validation, files_testing

    def _load_file(self, path, dtype=np.float32, padding=None):
        tile = 148  ##
        zer = np.zeros((tile, tile, tile), dtype=dtype)
        data = nrrd.read(path)[0].astype(dtype)
        zer = reshape_to_shape(data, (tile, tile, tile), padding)
        return zer

    def _next_data(self, datafile):
        if datafile == self.training_data_files:
            self.trainingfile_idx += 1
            if self.trainingfile_idx >= len(datafile):
                self.trainingfile_idx = 0
            image_name = datafile[self.trainingfile_idx]
        elif datafile == self.validation_data_files:
            self.validationfile_idx += 1
            if self.validationfile_idx >= len(datafile):
                self.validationfile_idx = 0
            image_name = datafile[self.validationfile_idx]
        elif datafile == self.testing_data_files:
            self.testingfile_idx += 1
            if self.testingfile_idx >= len(datafile):
                self.testingfile_idx = 0
            image_name = datafile[self.testingfile_idx]
        else:
            raise ValueError("Datafile Not Recognized")

        # logging.info("Case: {}".format(image_name))
        label_name = image_name.replace('case', 'needles')
        label_name = label_name.replace('_clahe', '')
        img = self._load_file(image_name, np.float32, padding="noise")
        label = self._load_file(label_name, np.uint16, padding="zero")

        return img, label
    
    def _next_data_label(self, datafile):
        if datafile == self.training_data_files:
            self.trainingfile_idx += 1
            if self.trainingfile_idx >= len(datafile):
                self.trainingfile_idx = 0
            image_name = datafile[self.trainingfile_idx]
        elif datafile == self.validation_data_files:
            self.validationfile_idx += 1
            if self.validationfile_idx >= len(datafile):
                self.validationfile_idx = 0
            image_name = datafile[self.validationfile_idx]
        elif datafile == self.testing_data_files:
            self.testingfile_idx += 1
            if self.testingfile_idx >= len(datafile):
                self.testingfile_idx = 0
            image_name = datafile[self.testingfile_idx]
        else:
            raise ValueError("Datafile Not Recognized")

        # logging.info("Case: {}".format(image_name))
        label_name = image_name.replace('case', 'needles')
        label_name = label_name.replace('_clahe', '')
        label = self._load_file(label_name, np.uint16, padding="zero")
        return label
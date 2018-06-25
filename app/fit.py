from fns import *
from unet import *

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'needlefinder')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

def main(argv):
    usage = ('usage: fit.py --InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath> --MinObjectSize <min obj size> --MaxLineFitError <min error size>')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputVolume', action='store', help='<InputVolumePath>', default='', dest='input')
    parser.add_argument('-o', '--OutputLabel', action='store', help='<OutputLabelPath>', default='', dest='output')
    parser.add_argument('-s', '--MinObjectSize', action='store', help='<min obj size>', default=100, dest='size')
    parser.add_argument('-e', '--MaxLineFitError', action='store', help='<min error size>', default=2.0, dest='error')
    # ModelName not used yet
    # parser.add_argument('-e', '--ModelName', action='store', help='<ModelName>', default='', dest='modelname')

    args = vars(parser.parse_args())
    print(args)
    InputVolume = args['input']
    OutputLabel = args['output']
    size = args['size']
    error = args['error']

    if InputVolume == '' or OutputLabel == '':
        print(usage)
        sys.exit()
    if os.path.isfile(InputVolume) and os.path.isdir(os.path.dirname(OutputLabel)):
        print("Making the model.")
        net = Unet(channels=1, 
           n_class=1, 
           layers=4, 
           pool_size=2,
           features_root=16, summaries=False,
          )
        print(50 * "-")
        print("Loading and Preparing the data")
        data, options = nrrd.read(InputVolume)
        data = data.astype(np.float32)
        arr_data = cutVolume(data)
        print(50 * "-")
        print("Starting the segmenter.")
        arr_pred = predict_full_volume(net, arr_data, model_path="/app/model/model.cpkt")
        print("Merging subvolumes")
        full_pred = recombine(arr_pred, data)
        islands = post_processing(full_pred, min_area=int(size), max_residual=float(error))
        nrrd.write(OutputLabel, islands, options=options)
    else:
        print("Make sure the input file exists and the output file directory is valid.")
        print("InputVolume: ", InputVolume)
        print("OutputLabel: ", OutputLabel)

if __name__ == "__main__":
    main(sys.argv[1:])

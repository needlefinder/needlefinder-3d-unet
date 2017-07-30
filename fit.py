from fns import *
from unet import *

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'needlefinder')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

def main(argv):

    parser = argparse.ArgumentParser(prog='fit.py')
    parser.add_argument('-i', '--InputVolume', nargs='+', help='<InputVolumePath>', dest='input', default='')
    parser.add_argument('-o', '--OutputLabel', nargs='+', help='<OutputLabelPath>', dest = 'output', default='')
    parser.add_argument('-s', '--size', nargs='?', help='<min obj size>', dest='size', default=100)
    parser.add_argument('-e', '--error', nargs='?', help='<min error size>', dest='error', default=2)

    InputVolume = parser.input
    OutputLabel = parser.output
    size = parser.size
    error = parser.error

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
        arr_pred = predict_full_volume(net, arr_data, model_path="/home/deepinfer/github/needlefinder-3d-unet/model/model.cpkt")
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

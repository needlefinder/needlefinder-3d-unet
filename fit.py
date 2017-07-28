from fns import *
from unet import *

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'needlefinder')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

def main(argv):
    InputVolume = ''
    OutputLabel = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["InputVolume=", "OutputLabel="])
    except getopt.GetoptError:
        print('usage: fit.py --InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fit.py --InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
            sys.exit()
        elif opt in ("-i", "--InputVolume"):
            InputVolume = arg
        elif opt in ("-o", "--OutputLabel"):
            OutputLabel = arg
    if InputVolume == '' or OutputLabel == '':
        print('usage: fit.py --InputVolume <InputVolumePath> -OutputLabel <OutputLabelPath>')
        sys.exit()
    if os.path.isfile(InputVolume) and os.path.isdir(os.path.dirname(OutputLabel)):
        ds = ProstateData()
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
	arr_pred = predict_full_volume(net, arr_data, model_path="./model/model.cpkt")
	print("Merging subvolumes")
	full_pred = recombine(arr_pred, data)
	nrrd.write(OutputLabel, full_pred, options=options)
    else:
        print("Make sure the input file exists and the output file directory is valid.")
        print("InputVolume: ", InputVolume)
        print("OutputLabel: ", OutputLabel)

if __name__ == "__main__":
    main(sys.argv[1:])

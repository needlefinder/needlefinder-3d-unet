from segment_in_two_steps import *
import os
import argparse
import sys

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'needlefinder')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

def main(argv):
    usage = ('usage: fit.py --InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputVolume', action='store', help='<InputVolumePath>', default='', dest='input')
    parser.add_argument('-o', '--OutputLabel', action='store', help='<OutputLabelPath>', default='', dest='output')
    args = vars(parser.parse_args())
    print(args)
    InputVolume = args['input']
    OutputLabel = args['output']

    if InputVolume == '' or OutputLabel == '':
        print(usage)
        sys.exit()
    if os.path.isfile(InputVolume) and os.path.isdir(os.path.dirname(OutputLabel)):
        segment_and_vote(InputVolume, OutputLabel)
    else:
        print("Make sure the input file exists and the output file directory is valid.")
        print("InputVolume: ", InputVolume)
        print("OutputLabel: ", OutputLabel)

if __name__ == "__main__":
    main(sys.argv[1:])

# 1. For each npz file, extract the words and write them to a sentence

import os
import shutil
import sys
import glob
import argparse
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract-cnn-features')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                        help='Folder to input files')
    parser.add_argument("-f", "--file-names", type=str, required=True,
                        help="""File containing a list with image file names.""")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output file')

    # Parse arguments
    args = parser.parse_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    file_names = Path(args.file_names).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    print("Output folder: %s" % str(output_file.parent))
    os.makedirs(output_file.parent, exist_ok=True)

    imglist = []
    with open(args.file_names) as f:
        for line in f.readlines():
            imglist.append(line.strip())

    with open(str(output_file), "w") as fd:
        for imgname in imglist:        
            input_file = input_folder / (os.path.splitext(imgname)[0] + ".npz")

            d = np.load(input_file)
            words = np.unique(d["objects"][d["objects_scores"] > 0.5])
            sentence = " ".join(words) + "\n"

            fd.write(sentence)

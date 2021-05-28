Image feature extraction.

## Installation

Tested with CUDA 10.2, Pytorch 1.7, Python 3.6.

Git clone and update submodules (`git clone $URL` and then `git submodule update --init --recursive`.

Create a new environment (`python3 -m venv env/` or `pipenv --python 3.6` if you are using pipenv) and activate the environment (`source env/bin/activate` or `pipenv shell`).

Install the requirements (`pip install requirements.txt` or `pipenv install`).

Install dependencies (`bash install.sh`).

Let `$ROOT` be the root directory of this repo. Create the `$ROOT/work` directory and cd to the directory.

## Multi30k dataset

Download the Mutlti30k dataset (from `https://github.com/vvjn/multi30k-wmt18`) into the `$ROOT/work/data/multi30k-wmt18` folder.
Raw images from Flickr30k can be found [here](https://forms.illinois.edu/sec/229675).
`test_2017_flickr` and `test_2018_flickr` images can be downloaded from [here](https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt).
Download the images into the `$ROOT/work/data/flickr30k-images` folder.

## Image feature extraction

Extract global image features using ResNet50. The pre-trained model is taken from (https://download.pytorch.org/models/resnet50-0676ba61.pth).

```
TEXT=data/multi30k-wmt18/task1-data
IMAGEDIR=data/flickr30k-images
MODELFILE=models/resnet50-0676ba61.pth
FEATSDIR=data/feats-flickr30k-images
TESTNAME=test_2016_flickr

python src/feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/train.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/train
python src/feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/val.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/valid
python src/feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/${TESTNAME}.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/test
```

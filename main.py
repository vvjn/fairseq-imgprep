import argparse
import os, sys
import os.path as osp
from fvcore.common.config import CfgNode
import subprocess

# Use cases:
# python main.py -d multi30k -i $multi30k_path -a preprocess_text -o $output_path
# python main.py -d multi30k -i $multi30k_path -a preprocess_mm -o $output_path

def joinpath(a,*p):
    return os.path.realpath(os.path.join(a, *p))

def run(C, cmd):
    print(" ".join(cmd))
    if not C.DRY_RUN: subprocess.run(cmd)

def preprocess_text_multi30k(C, prpath, cfgpath):
    cdir = os.getcwd()
    os.chdir(joinpath(cfgpath, C.DATASET.PATH_RAW))

    cmd = [
        "bash",
        "./scripts/task1-tokenize.sh"]
    run(C, cmd)

    cmd = [
        "bash",
        "./scripts/task1-tok-stats.sh"]
    run(C, cmd)

    cmd = [
        "bash",
        "./scripts/task1-bpe.sh",
        "-m", str(C.DATASET.BPE_MERGES)
    ]
    run(C, cmd)
    
    os.chdir(cdir)


def preprocess_mm_multi30k_resnet50(C, prpath, cfgpath):
    if C.SPLITS == "NONE":
        raise "Bad split"

    for split in C.SPLITS.split("+"):
        datasetnames = C.DATASET.get(split)
        imagesdirs = C.MULTIMODAL.RAW_DATA.get(split)
        featsdir = C.MULTIMODAL.DATA
        
        for name,imagesdir in zip(datasetnames, imagesdirs):

            if split == "TEST" and C.DATASET_NAME is not None and C.DATASET_NAME != name:
                print("Skipping dataset {}".format(name))
                continue

            if split in ["TRAIN","VALID"]:
                sname = split
            else:
                sname = name
            
            cmd = [
                "python",
                "{}/src/extract_image_feats_resnet50_indep.py".format(prpath),
                "-i", joinpath(cfgpath, imagesdir),
                "-f", joinpath(cfgpath, C.MULTIMODAL.SPLITS, name + ".txt"),
                "-m", joinpath(cfgpath, C.MULTIMODAL.MODEL_WEIGHTS),
                "-o", osp.dirname(joinpath(cfgpath, featsdir)),
                "-s", sname.lower(),
                "-b", str(C.MULTIMODAL.PREPROCESS_BATCH_SIZE)
            ]
            run(C, cmd)
    
def preprocess_mm_multi30k_vinvl(C, prpath, cfgpath):
    if C.SPLITS == "NONE":
        raise "Bad split"

    for split in C.SPLITS.split("+"):
        datasetnames = C.DATASET.get(split)
        imagesdirs = C.MULTIMODAL.RAW_DATA.get(split)
        featsdir = C.MULTIMODAL.DATA

        for name,imagesdir in zip(datasetnames, imagesdirs):

            if split == "TEST" and C.DATASET_NAME is not None and C.DATASET_NAME != name:
                print("Skipping dataset {}".format(name))
                continue

            if split in ["TRAIN","VALID"]:
                sname = split
            else:
                sname = name

            cmd = [
                "python",
                "{}/src/feats_vinvl.py".format(prpath),
                "--config-file", joinpath(cfgpath, C.MULTIMODAL.MODEL_PARAMS),                
                "--image-dir", joinpath(cfgpath, imagesdir),
                "--file-list", joinpath(cfgpath, C.MULTIMODAL.SPLITS, name + ".txt"),
                "--output-dir", joinpath(cfgpath, featsdir, "npz", name),
                "MODEL.WEIGHT", joinpath(cfgpath, C.MULTIMODAL.MODEL_WEIGHTS),
                "MODEL.ROI_HEADS.NMS_FILTER", "1", "MODEL.ROI_HEADS.SCORE_THRESH", "0.2", "TEST.IGNORE_BOX_REGRESSION", "False",
                "DATASETS.LABELMAP_FILE", joinpath(cfgpath, C.MULTIMODAL.OBJDET_LABELMAP_FILE)
            ]
            run(C, cmd)

            cmd = [
                "python", "{}/src/convert_vinvl_output_to_text.py".format(prpath),
                "--input-folder", joinpath(cfgpath, featsdir, "npz"),
                "--file-names", joinpath(cfgpath, featsdir, "npz", name + ".txt"),
                "--output-file", joinpath(cfgpath, featsdir, "text", name + ".vinvl.en"),
                "--threshold", str(C.MULTIMODAL.OBJDET_CONF_THRESH)
                ]
            run(C, cmd)

            cmd = [
                "bash", "{}/src/process-en/bpe-multi30k-task1.sh".format(prpath),
                "en", "vv",
                joinpath(cfgpath, C.DATASET.BPE_CODES),
                joinpath(cfgpath, C.DATASET.PATH, "vocab.en"),
                joinpath(cfgpath, featsdir, "text", name + ".vinvl.en"),
                joinpath(cfgpath, C.DATASET.PATH, name)
                ]
            run(C, cmd)

def preprocess_mm_multi30k_butd(C, prpath, cfgpath):
    if C.SPLITS == "NONE":
        raise "Bad split. Options: TRAIN,VALID,TEST"

    for split in C.SPLITS.split("+"):
        datasetnames = C.DATASET.get(split)
        imagesdirs = C.MULTIMODAL.RAW_DATA.get(split)
        featsdir = C.MULTIMODAL.DATA

        for name,imagesdir in zip(datasetnames, imagesdirs):

            print("split:", split, "DATASET_NAME:", C.DATASET_NAME)
            if split == "TEST" and C.DATASET_NAME is not None and C.DATASET_NAME != name:
                print("Skipping dataset {}".format(name))
                continue

            if split in ["TRAIN","VALID"]:
                sname = split
            else:
                sname = name

            cmd = [
                "python",
                "{}/src/feats_butd.py".format(prpath),
                "--mode", C.MULTIMODAL.BUTD.MODE,
                "--num-cpus", str(C.MULTIMODAL.PREPROCESS.NUM_CPUS),
                "--gpus", "'{}'".format(C.MULTIMODAL.BUTD.GPUS),
                "--extract-mode", C.MULTIMODAL.BUTD.EXTRACT_MODE,
                "--min-max-boxes", "'{},{}'".format(C.MULTIMODAL.BUTD.MIN_BOXES,C.MULTIMODAL.BUTD.MAX_BOXES),
                "--config-file", joinpath(cfgpath, C.MULTIMODAL.MODEL_PARAMS),                
                "--image-dir", joinpath(cfgpath, imagesdir),
                "--file-list", joinpath(cfgpath, C.MULTIMODAL.SPLITS, name + ".txt"),
                "--output-dir", joinpath(cfgpath, featsdir, "npz", name),
                "--objects-vocab", joinpath(prpath, "external/bottom-up-attention.pytorch/evaluation/objects_vocab.txt"),
                "--attributes-vocab", joinpath(prpath, "external/bottom-up-attention.pytorch/evaluation/attributes_vocab.txt"),
                "MODEL.WEIGHTS", joinpath(cfgpath, C.MULTIMODAL.MODEL_WEIGHTS)
            ]
            run(C, cmd)

            cmd = [
                "python", "{}/src/convert_butd_output_to_text.py".format(prpath),
                "--input-folder", joinpath(cfgpath, featsdir, "npz"),
                "--file-names", joinpath(cfgpath, featsdir, "npz", name + ".txt"),
                "--output-file", joinpath(cfgpath, featsdir, "text", name + ".butd.en"),
                "--threshold", str(C.MULTIMODAL.OBJDET_CONF_THRESH)
                ]
            run(C, cmd)

            cmd = [
                "bash", "{}/src/process-en/bpe-multi30k-task1.sh".format(prpath),
                "en", "vb",
                joinpath(cfgpath, C.DATASET.BPE_CODES),
                joinpath(cfgpath, C.DATASET.PATH, "vocab.en"),
                joinpath(cfgpath, featsdir, "text", name + ".butd.en"),
                joinpath(cfgpath, C.DATASET.PATH, name)
                ]
            run(C, cmd)

def main(C):
    prpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = C.ROOT_PATH

    if C.DATASET.NAME == "multi30k":
        if C.ACTION == "preprocess_text":
            preprocess_text_multi30k(C, prpath, cfgpath)

        elif C.ACTION == "preprocess_mm":

            if C.MULTIMODAL.TYPE == "resnet50":
                preprocess_mm_multi30k_resnet50(C, prpath, cfgpath)

            elif C.MULTIMODAL.TYPE == "vinvl":
                preprocess_mm_multi30k_vinvl(C, prpath, cfgpath)

            elif C.MULTIMODAL.TYPE == "butd":
                preprocess_mm_multi30k_butd(C, prpath, cfgpath)

            else:
                print("Unknown mutimodal type.")
                return

        else:
            print("Unknown action.")
            return

    else:
        print("Unknown dataset.")
        return

    print("Done.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", help="path to config file")
    parser.add_argument(
        "-a", "--action", help="action", type=str, required=True)
    parser.add_argument(
        "-s", "--split", help="split (none,train,valid,test,train+valid,etc)", type=str, default="none")
    parser.add_argument(
        "-d", "--dataset-name", help="dataset name", type=str, default=None)
    parser.add_argument(
        "-n", "--dry-run", help="dry run", action="store_true")
    parser.add_argument(
        "opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    C = CfgNode(new_allowed=True)
    C.ROOT_PATH = os.path.dirname(os.path.realpath(args.config_file))
    C.VERBOSE = False
    C.DRY_RUN = False
    C.merge_from_file(args.config_file)
    C.merge_from_list(args.opts)
    C.ACTION = args.action
    C.SPLITS = args.split.upper()
    if args.dry_run:
        C.DRY_RUN = True
    C.DATASET_NAME = args.dataset_name
    C.freeze()

    print("With base path {}".format(C.ROOT_PATH))
    
    main(C)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features, save_scores_and_roi_features
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes

import ray
from ray.actor import ActorHandle

from pathlib import Path

def switch_extract_mode(mode):
    if mode in ['roi_feats','roi_feats_and_confs']:
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def generate_npz(extract_mode, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    elif extract_mode == 4:
        save_scores_and_roi_features(*args)
    else:
        print('Invalid Extract Mode! ')

@ray.remote(num_gpus=1)
def extract_feat(split_idx, img_list, cfg, args, actor: ActorHandle):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    for im_file in (img_list):
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            actor.update.remote(1)
            continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
            actor.update.remote(1)
            continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]

            if args.extract_mode == "roi_feats":
                generate_npz(1, args, cfg, im_file, im, dataset_dict, 
                             boxes, scores, features_pooled, attr_scores)
            elif args.extract_mode == "roi_feats_and_confs":
                generate_npz(4, args, cfg, im_file, im, dataset_dict, 
                             boxes, scores, features_pooled, attr_scores)
            else:
                raise(ValueError("args.extract_mode"))
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz(2,
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores)
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
                actor.update.remote(1)
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
            generate_npz(3, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores, features_pooled, attr_scores)

        actor.update.remote(1)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes', 'bbox_feats', 'roi_feats_and_confs' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively', 'extract roi features and confidence scores directly'")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--output-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")

    parser.add_argument('--file-list', help="list of image file names")    

    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")

    parser.add_argument('--objects-vocab', dest='objects_vocab',
                        help='file objects vocab')
    parser.add_argument('--attributes-vocab', dest='attributes_vocab',
                        help='file attributes vocab')
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = setup(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Load classes
    classes = [] # ['__background__']
    with open(args.objects_vocab) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    # Load attributes
    attributes = [] # ['__no_attribute__']
    with open(args.attributes_vocab) as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract features.
    imglist = []
    with Path(args.file_list).open() as f:
        for fname in f:
            afname = Path(args.image_dir) / fname.strip()
            assert afname.exists(), "{} does not exist.".format(afname)
            imglist.append(fname.strip())

    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    if num_images == 0:
        exit()

    npsfile = args.output_dir
    npsfile = npsfile[:-1] if npsfile.endswith("/") else npsfile
    base_dir = os.path.basename(npsfile)
    npsfile = npsfile + ".txt"
    if os.path.exists(npsfile):
        print("File exists {}".format(npsfile))
        exit()
    fd_list = open(npsfile, "w")
    for imfile in imglist:
        ofile = base_dir + "/" + os.path.splitext(imfile)[0] + ".npz\n"
        fd_list.write(ofile)
    fd_list.close()

    # Save class and attribute names to file
    np.savez_compressed(
        os.path.splitext(npsfile)[0] + "_info.npz",
        classes=classes, attributes=attributes, cfg=cfg, args=args) #info={'cfg':cfg, 'args':args})

    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

    pb = ProgressBar(len(imglist))
    actor = pb.actor

    print('Number of GPUs: {}.'.format(num_gpus))
    extract_feat_list = []
    for i in range(num_gpus):
        extract_feat_list.append(extract_feat.remote(i, img_lists[i], cfg, args, actor))
    
    pb.print_until_done()
    ray.get(extract_feat_list)
    ray.get(actor.get_counter.remote())

if __name__ == "__main__":
    main()

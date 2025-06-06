# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import os.path as op
import argparse
import json
import numpy as np
import os
from pathlib import Path

from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from tools.demo.detect_utils import detect_objects_on_single_image
from tools.demo.visual_utils import draw_bb, draw_rel


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config-file", metavar="FILE",
                        help="path to config file")

    parser.add_argument("--output-dir", help="output directory for features")
    parser.add_argument("--image-dir", help="input directory with images")
    parser.add_argument("--file-list", help="list of image files")
    
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    mkdir(args.output_dir)

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    visual_labelmap = None

    if cfg.MODEL.ATTRIBUTE_ON:
        dataset_attr_labelmap = {
            int(val): key for key, val in
            dataset_allmap['attribute_to_idx'].items()}
    
    if cfg.MODEL.RELATION_ON:
        dataset_relation_labelmap = {
            int(val): key for key, val in
            dataset_allmap['predicate_to_idx'].items()}

    transforms = build_transforms(cfg, is_train=False)

    imglist = []
    with Path(args.file_list).open() as f:
        for fname in f:
            afname = Path(args.image_dir) / fname.strip()
            assert afname.exists(), "{} does not exist.".format(afname)
            imglist.append(fname)
    
    num_images = len(imglist)
    print("Number of images: {}.".format(num_images))

    fd_list = open(args.output_dir + ".txt", "w")
    listdir = op.basename(args.output_dir)

    for imgfile in imglist:

        image_file = op.join(args.image_dir, imgfile)
        npz_file = op.join(args.output_dir, op.splitext(imgfile)[0] + ".npz")
        fd_list.write(listdir + "/" + op.splitext(imgfile)[0] + ".npz\n")
    
        assert op.isfile(image_file), \
            "Image: {} does not exist".format(image_file)

        if op.isfile(npz_file):
            print("Output exists: {}. Skipping.".format(npz_file))
            continue

        cv2_img = cv2.imread(image_file)

        if cv2_img is None:
            print("Bad image file: {}. Skipping.".format(image_file))
            continue

        print("Processing {}.".format(image_file))

        dets = detect_objects_on_single_image(model, transforms, cv2_img)

        if isinstance(model, SceneParser):
            rel_dets = dets['relations']
            dets = dets['objects']
    
        for obj in dets:
            obj["class"] = dataset_labelmap[obj["class"]]
        if visual_labelmap is not None:
            dets = [d for d in dets if d['class'] in visual_labelmap]
        if cfg.MODEL.ATTRIBUTE_ON:
            for obj in dets:
                obj["attr"], obj["attr_conf"] = postprocess_attr(dataset_attr_labelmap, obj["attr"], obj["attr_conf"])
    
        if cfg.MODEL.RELATION_ON:
            for rel in rel_dets:
                rel['class'] = dataset_relation_labelmap[rel['class']]
                subj_rect = dets[rel['subj_id']]['rect']
                rel['subj_center'] = [(subj_rect[0]+subj_rect[2])/2, (subj_rect[1]+subj_rect[3])/2]
                obj_rect = dets[rel['obj_id']]['rect']
                rel['obj_center'] = [(obj_rect[0]+obj_rect[2])/2, (obj_rect[1]+obj_rect[3])/2]
    
        rects = [d["rect"] for d in dets]
        scores = [d["conf"] for d in dets]
        if cfg.MODEL.ATTRIBUTE_ON:
            attr_labels = [' '.join(d["attr"]) for d in dets]
            attr_scores = [d["attr_conf"] for d in dets]
            labels = [attr_label+' '+d["class"]
                      for d, attr_label in zip(dets, attr_labels)]
        else:
            labels = [d["class"] for d in dets]
    
        #print("IMGFILE:", args.img_file, op.splitext(args.img_file))
        #print("OUTPUT:", args.output_dir)
        #print("ATTR vis:", cfg.MODEL.ATTRIBUTE_ON)

        if cfg.MODEL.ATTRIBUTE_ON:
            np.savez_compressed(
                npz_file,
                rects=rects,
                objects_scores=scores,
                objects=[d["class"] for d in dets],
                attrs_scores = attr_scores,
                attrs = attr_labels            
            )
        else:
            np.savez_compressed(
                npz_file,
                rects=rects,
                objects_scores=scores,
                objects=[d["class"] for d in dets]
            )
    
        #draw_bb(cv2_img, rects, labels, scores)
    
        # if cfg.MODEL.RELATION_ON and args.visualize_relation:
        #     rel_subj_centers = [r['subj_center'] for r in rel_dets]
        #     rel_obj_centers = [r['obj_center'] for r in rel_dets]
        #     rel_scores = [r['conf'] for r in rel_dets]
        #     rel_labels = [r['class'] for r in rel_dets]
        #     draw_rel(cv2_img, rel_subj_centers, rel_obj_centers, rel_labels, rel_scores)
    
        # if not args.save_file:
        #     save_file = op.splitext(args.img_file)[0] + ".detect.jpg"
        # else:
        #     save_file = args.save_file
        # cv2.imwrite(save_file, cv2_img)
        # print("save results to: {}".format(save_file))
    
        # # save results in text
        # if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        #     result_str = ""
        #     for label, score, attr_score in zip(labels, scores, attr_scores):
        #         result_str += label+'\n'
        #         result_str += ','.join([str(conf) for conf in attr_score])
        #         result_str += '\t'+str(score)+'\n'
        #     text_save_file = op.splitext(save_file)[0] + '.txt'
        #     with open(text_save_file, "w") as fid:
        #         fid.write(result_str)

    fd_list.close()
    

if __name__ == "__main__":
    main()

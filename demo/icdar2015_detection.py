# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="./configs/ocr/icdar2015_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default="./out_dir_r101/icdar2015_model/model_ic15_r101.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default="./input_images",
        nargs="+",
        help="image or folder of icdar2015 images"
    )

    parser.add_argument(
        "--output",
        default="./test_icdar2015/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def compute_polygon_area(points):
    s = 0
    point_num = len(points)
    if(point_num < 3): return 0.0
    for i in range(point_num): 
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)
    

def save_result_to_txt(txt_save_path,prediction,polygons):

    file = open(txt_save_path,'w')

    classes = prediction['instances'].pred_classes

    for i in range(len(classes)):
        if classes[i]==0:
            if len(polygons[i]) != 0:
                points = []
                for j in range(0,len(polygons[i][0]),2):
                    points.append([polygons[i][0][j],polygons[i][0][j+1]])
                points = np.array(points)
                area = compute_polygon_area(points)
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)

                if area > 175:
                    file.writelines(str(int(box[0][0]))+','+str(int(box[0][1]))+','+str(int(box[1][0]))+','+str(int(box[1][1]))+','
                              +str(int(box[2][0]))+','+str(int(box[2][1]))+','+str(int(box[3][0]))+','+str(int(box[3][1])))
                    file.write('\r\n')

    file.close()


if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = setup_cfg(args)
    detection_demo = VisualizationDemo(cfg)

    test_images_path = args.input
    output_path = args.output

    start_time_all = time.time()
    img_count = 0
    for i in glob.glob(test_images_path):
        print(i)
        img_name = os.path.basename(i)
        img_save_path = output_path + img_name.split('.')[0] + '.jpg'
        img = cv2.imread(i)
        start_time = time.time()

        prediction, vis_output, polygons = detection_demo.run_on_image(img)

        txt_save_path = output_path + 'res_' + img_name.split('.')[0] + '.txt'
        save_result_to_txt(txt_save_path,prediction,polygons)

        print("Time: {:.2f} s / img".format(time.time() - start_time))
        vis_output.save(img_save_path)
        img_count += 1
    print("Average Time: {:.2f} s /img".format((time.time() - start_time_all) / img_count))



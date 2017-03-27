#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
matplotlib.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# imagenet detection 200 categories
CLASSES = ('__background__', 'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam', 'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap', 'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle', 'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker', 'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew', 'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper', 'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly', 'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig', 'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog', 'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart', 'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger', 'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim', 'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse', 'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle', 'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can', 'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace', 'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume', 'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza', 'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine', 'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse', 'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator', 'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler', 'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine', 'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie', 'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet', 'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin', 'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft', 'whale', 'wine_bottle', 'zebra')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = './models/imagenet/vgg16/faster_rcnn_end2end/text.prototxt'
    caffemodel = '/disk2/data/ILSVRC2015/output/faster_rcnn_end2end/ILSVRC_2015_train/vgg16_faster_rcnn_iter_100000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
        savepath="imagenet_detection_"+im_name
        plt.savefig(savepath)

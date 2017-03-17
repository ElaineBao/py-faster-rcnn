# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class imagenet(imdb):
    def __init__(self, image_set, year, ILSVRC_path= None):
        imdb.__init__(self, 'ILSVRC_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._ILSVRC_path = cfg.ROOT_DIR if ILSVRC_path is None \
                            else ILSVRC_path
        self._map_det_path = os.path.join(self._ILSVRC_path, 'devkit/data/map_det.txt')
        self._DET_path = os.path.join(self._ILSVRC_path,'DET')
        self._classes, self._class_to_ind, self._class_to_name \
                        = self._load_class_wnids(self._map_det_path)
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.rpn_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._ILSVRC_path), \
                'Path does not exist: {}'.format(self._ILSVRC_path)
        assert os.path.exists(self._map_det_path), \
                'Path does not exist: {}'.format(self._map_det_path)
        assert os.path.exists(self._DET_path), \
                'Path does not exist: {}'.format(self._DET_path)

    def _load_class_wnids(self, map_det_path):
        classes = ['__background__'] # always index 0
        classes_to_ind = {}  # key = WNID, value = ILSVRC2015_DET_ID
        classes_to_name = {}  # key = WNID, value = class name
        for line in open(map_det_path):
            WNID, ILSVRC2015_DET_ID, class_name = line.split(' ', 2)
            classes.append(WNID)
            classes_to_ind[WNID] = ILSVRC2015_DET_ID
            classes_to_name[WNID] = class_name

        return tuple(classes), classes_to_ind, classes_to_name


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._DET_path, 'Data', 'DET', self._image_set,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._DET_path + /ImageSets/DET/train_1.txt
        image_set_file = os.path.join(self._DET_path, 'ImageSets', 'DET')
        if self._image_set == 'train': 
            image_index = []
            for i in xrange(1,201):  # there are 200 image_set_file for training
                i_image_set_file = os.path.join(image_set_file, \
                                                self._image_set + '_' + str(i) + '.txt')
                assert os.path.exists(i_image_set_file), \
                    'Path does not exist: {}'.format(i_image_set_file)
                with open(i_image_set_file) as f:
                    image_index.extend([x.split(' ')[0] for x in f.readlines()])
        else:
            image_set_file = os.path.join(self._DET_path, 'ImageSets', 'DET',\
                                          self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index = [x.strip() for x in f.readlines()]
        
        return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_ILSVRC_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_ILSVRC_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the ILSVRC
        format.
        """
        filename = os.path.join(self._DET_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


if __name__ == '__main__':
    from datasets.imagenet import imagenet
    d = imagenet('train', '2015', '/disk2/data/ILSVRC2015')
    res = d.roidb
    from IPython import embed; embed()

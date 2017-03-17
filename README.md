# Faster R-CNN with Imagenet dataset

This repository uses the python implementation of faster-rcnn [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) to train on imagenet dataset.

### Difference

Most of the code is similar to the original py-faster-rcnn, expect the following scripts:  

(We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`)

1. **$FRCN_ROOT/lib/datasets/imagenet.py & factory.py**: add imagenet data interface.  

The original py-faster-rcnn is trained on pascal_voc / coco dataset, there is no data interface for imagenet. So I add one for imagenet.

2. **$FRCN_ROOT/models/imagenet**: add models for imagenet.  

Again the original py-faster-rcnn only provides prototxt for pascal_voc / coco dataset. The model I use for training imagenet is VGG16.

3. **$FRCN_ROOT/experiments/cfgs/imagenet_faster_rcnn_end2end.yml**: add imagenet config file. This should base on your path to imagenet dataset.

4. **$FRCN_ROOT/experiments/scripts/faster_rcnn_end2end.sh**: add imagenet train shell command.

### To train with imagenet

1. first, download the training data and imagenet devkit  

I use the training data from ILSVRC 2015, and arrange the data folder structure as follows:

>|---ILSVRC  
 >>|---DET  
    >>>|---Annotations  
       >>>>|---DET  
          >>>>>|---train  
             >>>>>>|---ILSVRC2013_train  
                >>>>>>>|---n04272054  
                  >>>>>>>>|---n04272054_*.xml  
                  >>>>>>>>|---...  
             >>>>>>|---...  
          >>>>>|---val  
    >>>|---Data  
        >>>>|---DET  
          >>>>>|---train  
             >>>>>>|---ILSVRC2013_train  
                >>>>>>>|---n04272054  
                  >>>>>>>>|---n04272054_*.JPEG  
                  >>>>>>>>|---...  
             >>>>>>|---...  
          >>>>>|---val          
    >>>|---ImageSets  
        >>>>|---DET  
          >>>>>|---train_1.txt  
          >>>>>|---train_2.txt  
          >>>>>|---...  
          >>>>>|---train_200.txt  
          >>>>>|---val.txt  
 >>|---devkit  
    >>>|---data  
       >>>>|---map_det.txt  


2. Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

3. Modify config file

The file locates in $FRCN_ROOT/experiments/cfgs/imagenet_faster_rcnn_end2end.yml,   
you can set path to imagenet dataset, etc. in this file.

4. Train faster-rcnn with Imagenet dataset

There are two training algorithms provided by the faster-rcnn NIPS 2015 paper, one is **alternating optimization** algorithm, 
and the other is **approximate joint training** algorithm.  

Here I use the approximate joint training algorithm, as it resulting in faster (~ 1.5x speedup) training times and similar detection accuracy.  

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [DATASET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# DATASET is imagenet, or you can also use pascal voc, coco dataset.
# --set ... allows you to specify fast_rcnn.config options, e.g.
```


For other issues, please visit the original [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
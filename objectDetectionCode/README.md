# Object detection reference training + _all python scripts_

---
### _Training:_
This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### RetinaNet
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

---
### train .py
This file is dedicated to the traning phase of a model. You can start from scratch or save the model for each epoch. You can also evalaute the model once in a while or for each iteration. Several parameters and hyperparameters can be specified. 

---
### engine .py
Inside this script is called every training epoch and manages from a "lower level" the single training phase. Several options are related to key-points detection and segmentation but in this project only _iou_types = ["bbox"]_ is considered (object detection, square bounding boxes).
This script also contains the evaluate function, needed to evaluate the model and print metrices.

---
### detect .py
This script is used to perform a qualitative object detection task. You can decide the difficulty of the testing subset from which _"--num-images"_ images are randomly picked.

---
### evaluatePerDifficulty .py
This script allows to evaluate the final or a resumed model on a specific subset of the training set (easy/medium/hard). Code mainly comes from train .py. 

---
### presets .py
This script constains some (basic) default transformation that are used for image augmentation purposes.


---
### transforms .py
This script contains some functional torchvision transformations that are called inside presets .py.


---
### utils .py
This script contains utility functions that are called inside several other scripts.

---
### coco_utils .py, coco_subset .py, coco_eval .py, coco_names.py
These are all files coming from the default coco libraries with some slight changes. 

---
### group_by_aspect_ratio.py
It contains utility functions for batch images.

---
### AVD_to_COCO.py
This script converts AVD annotations into MSCOCO format and combines annotations from multiple scenes into a single file.

---
### transforms_AVD.py
This script isn't really used because I "translated" the AVD annotations into COCO format in order to use the "stock" COCO code. Inside this transforms script there are some potentially useful functions to (also) perform data augmentation, if AVD format was kept from the beginning. For instance: AddBackgroundBoxes, ResizeImage, AddRandomBoxes, BlackOutObjects.
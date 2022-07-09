'''
This script is used to perform a qualitative object detection task. 
You can decide the difficulty of the testing subset from which "--num-images" 
images are randomly picked.
'''
import datetime
import os
import time
import json
import cv2
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import numpy as np

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils
import random

from pathlib import Path
def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)

    return (b,g,r)



def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 34),     #AVD classes are 34 (33+background) not 91
        "coco_kp": (data_path, get_coco_kp, 2) #not used in this project (key-point detection)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()

#Function to extract a random image's name from the annotation files (not empty)
def randomImageName(TEST_PATH, ANN_PATH):
    import json
    from random import seed
    from random import randint   

    annFile = open(ANN_PATH,)    
    data = json.load(annFile)
    imageName = ''
    randNumber = randint(0, len(data['images'])-1)    
    cont = -1
    for i in data['images']: 
        cont += 1  
        if(cont == randNumber):
            imageName = i['file_name']
              
    #Check if it's an empty image
    cont = -1    
    for k in data['annotations']:
        cont += 1 
        if(cont == randNumber & len(k['bbox']) < 1):
            imageName = "None"
            return imageName


    annFile.close()
    imagePath = os.path.join(TEST_PATH, imageName)
    print("Selected image: ", imageName)
    return imagePath

def main(args):
      
    num_classes = 34 #modified for AVD   

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    COCO_ROOT=Path('./google_drive/MyDrive/DL_Project/')
    winname = 'result'

    #Commented because I'm running on Colab (not local) and it can't show images this way
    #cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(winname, width=640, height=480)
   
    device = torch.device('cpu')  
    kwargs = {
        "trainable_backbone_layers": 2
    }
    model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=34,
                                                                             pretrained=False,
                                                                             **kwargs)
                                                             
    model.to(device) #this detection code can be run on CPU
    model_without_ddp = model

    #model parameters (same as the trained model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
    
    if args.checkpoint_model: #load checkpoint model
        checkpoint = torch.load(args.checkpoint_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    elif args.trained_model: #load fully trained model
      model.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu'))['model'])

    #model.cuda() #this detection code can be run on CPU
    model.eval()

    #Difficulty and related testset
    print("\nSelected difficulty: ", args.difficulty)  

    DATA_PATH = 'google_drive/MyDrive/DL_Project/'
    if(args.difficulty == 'easy'):
      TEST_PATH = os.path.join(DATA_PATH, "valAVD/easy")
      ANN_PATH = os.path.join(DATA_PATH, "annotations/instances_easy_valAVD.json")
      dataset_test, _ = get_dataset('coco', "val_easy", get_transform(train=False), DATA_PATH)
    elif(args.difficulty == 'medium'):
      TEST_PATH = TEST_PATH = os.path.join(DATA_PATH, "valAVD/medium")
      ANN_PATH = os.path.join(DATA_PATH, "annotations/instances_medium_valAVD.json")
      dataset_test, _ = get_dataset('coco', "val_medium", get_transform(train=False), DATA_PATH)
    elif(args.difficulty == 'hard'):
      TEST_PATH = os.path.join(DATA_PATH, "valAVD/hard")
      ANN_PATH = os.path.join(DATA_PATH, "annotations/instances_hard_valAVD.json")
      dataset_test, _ = get_dataset('coco', "val_hard", get_transform(train=False), DATA_PATH)
            
    #Extract "num_images" names from the right annotation file
    numImages = args.num_images

    for i in range(numImages):
        #Looking for images with objects
        im_name = "None"
        while im_name == "None":
            im_name = randomImageName(TEST_PATH, ANN_PATH)

        src_img = cv2.imread(im_name)
        src_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(src_img/255.).permute(2,0,1).float()#.cuda()
        input = []
        input.append(img_tensor)
        out = model(input)        #trained model output
        boxes = out[0]['boxes']   #predicted boxes
        labels = out[0]['labels'] #predicted labels
        scores = out[0]['scores'] #predicted scores
        
        #Retrieve AVD dataset categories
        AVD_root_path = "./google_drive/MyDrive/DL_Project/annotations/"
        map_file = open(os.path.join(AVD_root_path,'instancesMapping.txt'),'r')
        categories = []
        
        for line in map_file:
            line = str.split(line)
            cid = int(line[1])
            name = line[0]
            categories.append({'id':cid, 'name':name})
  

        #Important: labels[k]-1 is used to compensate the background class
        print("Scores of the detected objects:")
        for k in range(len(scores)):
          print("Predicted object: ", categories[labels[k] - 1]['name'],
                " with score: ", scores[k].detach().numpy())
        print("\n")
        for idx in range(boxes.shape[0]):
            if scores[idx] >= args.threshold:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                
                # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
                '''
                x1 = int(x1.detach().cuda().detach().cpu().clone().numpy().astype(np.float32))
                x2 = int(x2.detach().cuda().detach().cpu().clone().numpy().astype(np.float32))
                y1 = int(y1.detach().cuda().detach().cpu().clone().numpy().astype(np.float32))
                y2 = int(y2.detach().cuda().detach().cpu().clone().numpy().astype(np.float32))
                '''
                #CPU friendly code:
                x1 = int(x1.detach().clone().numpy().astype(np.float32))
                x2 = int(x2.detach().clone().numpy().astype(np.float32))
                y1 = int(y1.detach().clone().numpy().astype(np.float32))
                y2 = int(y2.detach().clone().numpy().astype(np.float32))
                
                name = str(categories[labels[idx] - 1]['name'])
                cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

        #cv2.imshow(winname, src_img)  

        #save the image instead of showing it        
        detImgName = "detectedImage_" + str(i+1) + ".jpg"         
        cv2.imwrite(detImgName, src_img)
        print("Successfully saved image", i+1, "!\n\n")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument("--difficulty", default = 'easy',help="Choose the difficulty of testSet")
    parser.add_argument('--trained-model', default='', help='model loaded from finished training')
    parser.add_argument('--checkpoint-model', default='', help='model loaded from checkpoint')
    parser.add_argument('--threshold', default=0.5,type=float, help='minimum score to show the bbox')    
    parser.add_argument('--num-images', default=1, type=int, help='number of images to save')    
    
    args = parser.parse_args()
    main(args)

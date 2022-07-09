'''
This script converts AVD annotations into MSCOCO format.
Combines annotations from multiple scenes into a single file that can be used
with MSCOCO compatible code.
'''

import os
import json


ANNOTATIONS_PATH = "./google_drive/MyDrive/DL_Project/annotations_AVD_structure"
#Not necessary paths, need to be modified depending on the data structure
#IMAGES_PATH_TRAIN = "./google_drive/MyDrive/DL_Project/trainDataset_originalAVD"
#IMAGES_PATH_TEST = "./google_drive/MyDrive/DL_Project/testDataset_originalAVD"
save_path = './'

#Default train and validation/test scene lists
train_scene_list = ['Home_001_1',
                    'Home_002_1',
                    'Home_003_1',
                    'Home_004_1',             
                    'Home_005_1',
                    'Home_006_1',
                    'Home_007_1',
                    'Home_008_1',
                    'Home_010_1',
                    'Home_011_1',
                    'Home_014_1',
                    'Office_001_1'
                    ]

test_easy_scene_list =   ['Home_005_2', 'Home_015_1']
test_medium_scene_list = ['Home_001_2', 'Home_016_1', 'Home_014_2']
test_hard_scene_list =   ['Home_003_2', 'Home_004_2', 'Home_013_1']


#Scene list, train or test selection
SCENE_PATH = ANNOTATIONS_PATH #IMAGES_PATH_TEST #IMAGES_PATH_TRAIN 
scene_list = test_medium_scene_list
save_name = 'instances_medium_valAVD.json'

if not(os.path.isdir(save_path)):
    os.makedirs(save_path)

#map_file contains the correspondences between category names and ids
map_file = open(os.path.join(ANNOTATIONS_PATH,'instancesMapping.txt'),'r')
categories = []

#split the line and append category id and name
for line in map_file:
    line = str.split(line)
    cid = int(line[1])
    name = line[0]
    categories.append({'id':cid, 'name':name})

img_anns = []
box_anns = []
cids = []
box_id_counter = 0

#To browse each scene folder with the respective annotation.json file:
for scene in scene_list:
    scene_path = os.path.join(SCENE_PATH,scene)
    annotations = json.load(open(os.path.join(scene_path,'annotations.json')))

    for img_name in annotations.keys():

        img_ind = int(img_name[:-4])

        pre_boxid_counter = box_id_counter
        boxes = annotations[img_name]['bounding_boxes']
   
        #if(len(boxes) > 0): #used to "log" only images with objects, not used in general
        for box in boxes:              
            xmin = box[0]
            ymin = box[1]
            width = box[2]-box[0] +1
            height = box[3]-box[1] +1
            iscrowd = 0
            if max(width, height) <= 25 or min(width,height) <= 15:
              iscrowd=1
            if box[5] >= 5:
                iscrowd=1

            area = width*height
            cid = box[4]
          
            cids.append(cid)  

            ##Equivalent notation for segmentations
            #x1, x2, y1, y2 = [box[0], box[0]+box[2], box[1], box[1]+box[3]] 
            #[[x1, y1, x1, y2, x2, y2, x2, y1]]  

            #more clear notation to me:
            x1 = xmin
            y1 = ymin
            x2 = box[2]
            y2 = box[3]               
            box_anns.append({'area':area,'bbox':[xmin,ymin,width,height],
                            'category_id':cid,'image_id':img_ind,
                            'iscrowd':iscrowd, 'segmentation': [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]],        
                            'id':box_id_counter})
            box_id_counter += 1
        #AVD images are kept full resolution
        img_anns.append({'file_name':img_name, 'id':img_ind, 'height':1080, 'width':1920})
coco_anns = {'images':img_anns, 'annotations':box_anns,'categories':categories}

with open(os.path.join(save_path,save_name), 'w') as outfile:
    json.dump(coco_anns, outfile)
    print("Annotation file successfully written")
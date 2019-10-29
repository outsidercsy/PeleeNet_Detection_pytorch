import torch
import torch.nn as nn
import  numpy as np
import cv2
import os
import os.path as osp
import datetime
import matplotlib.pyplot as plt
import json
import glob
import pickle



################gene img with  gt  bbox
# category_id_map = {'40': 0, '43': 1, '44': 2, '45': 3, '46': 4, '47': 5, '49': 6, '50': 7, '51': 8, '54': 9, '57': 10, '59': 11, '60': 12, '61': 13}

# os.chdir('/Users/outsider/Desktop/holo')

# categories = set()

# _detections={}
# _image_ids=[]



# ####left
# _image_dir = './left/1129_left_image_annotation_4'
# _anno_dir = './left/result/1129_left_image_annotation_4'


# # _image_dir = './right/1127_right_image_annotation_3'
# # _anno_dir = './right/result/1127_right_image_annotation_3'


# image_names = glob.glob(os.path.join(_image_dir, '*.jpeg'))
# image_names.sort()




# for image_name in image_names:

#     image_name = image_name[-13:]

#     if image_name[-4:] != 'jpeg':
#         print('not match')
#         continue

    
#     annos = []
#     if not os.path.exists(osp.join(_anno_dir, image_name.rstrip('.jpeg') + '.txt')):
#         print('not exist image_name', image_name)
#         continue
#     with open( osp.join(_anno_dir, image_name.rstrip('.jpeg') + '.txt'), 'r' ) as f:
#         js = f.read()


#     js = json.loads(js)   
    
#     for item in js:
  

#         annos.append(item['points'] + [  item['category']]   )
#         categories.add(item['category'])
        
#     if len(annos) == 0:
#         print('without annos')
#         continue
#     annos = np.array(annos, dtype=np.float32)

#     _image_ids.append(image_name)
#     _detections[image_name] = annos

#     # img = cv2.imread(osp.join(_image_dir, image_name))

#     # height, width, _ = img.shape 

    
#     # for anno in annos:
        
#     #     if (anno[2] - anno[0]) * (anno[3] - anno[1]) < 0.005 * width * height:
#     #         color = [0, 0, 255]####r
#     #     else:
#     #         color = [0, 255, 0]####g
#     #     cv2.rectangle(img,
#     #         (anno[0], anno[1]),
#     #         (anno[2], anno[3]),
#     #         color, 1
#     #     )
#     #     cv2.putText(img, str(anno[4]), 
#     #         (anno[0], anno[1]), 
#     #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1
#     #     )



#     # cv2.imwrite('/Users/outsider/Desktop/holo_img_with_annos/' + image_name, img) #########

#################





############statistic of holo
# os.chdir('/Users/outsider/Desktop/holo')


# category_id_map = {'40': 0, '43': 1, '44': 2, '45': 3, '46': 4, '47': 5, '49': 6, '50': 7, '51': 8, '54': 9, '57': 10, '59': 11, '60': 12, '61': 13}


# category_object_cnt = {'40': 0, '43': 0, '44': 0, '45': 0, '46': 0, '47': 0, '49': 0, '50': 0, '51': 0, '54': 0, '57': 0, '59': 0, '60': 0, '61': 0}


# _detections={}
# _image_ids=[]



# _image_dir = './images'
# _anno_dir = './annos'


# image_names = glob.glob(os.path.join(_image_dir, '*.jpeg'))


# image_names.sort()

# import pdb
# pdb.set_trace()


# for image_name in image_names:

#     image_name = image_name[-13:]

#     if image_name[-4:] != 'jpeg':
#         print('not match')
#         continue

    
#     annos = []
#     if not os.path.exists(osp.join(_anno_dir, image_name.rstrip('.jpeg') + '.txt')):
#         print('not exist image_name', image_name)
#         continue
#     with open( osp.join(_anno_dir, image_name.rstrip('.jpeg') + '.txt'), 'r' ) as f:
#         js = f.read()


#     js = json.loads(js)   
    
#     for item in js:
  

#         annos.append(item['points'] + [  item['category']]   )

#         category_object_cnt[item['category']]+=1

#     if len(annos) == 0:
#         print('without annos')
#         continue
#     annos = np.array(annos, dtype=np.float32)

#     _image_ids.append(image_name)
#     _detections[image_name] = annos

#################



def test():
    for i in range(10):
        print(i)
        continue
        return 0


a = test()
print(a)

import os
import json
import pprint
import shutil
import numpy as np
from PIL import Image

json_path = './annotations_trainval2014/annotations/'
image_path = './val2014/'
copy_path = './my_test/'

# Read coco format 
with open(json_path+'instances_val2014.json', 'r') as f:
    coco_instance_json = json.load(f)


def parse_segmentation(segmentation):
    return np.array(segmentation).reshape(-1,2).tolist()


def parse_bbox(bbox):
    """ return rectangle"""
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    
    return [[x, y], [x+w, y+h]]


def file_name(id):
    return f'COCO_val2014_{str(id).zfill(12)}'


def get_size(image_path):
    img = Image.open(image_path)
    width, height = img.size
    return height, width


def get_annotation(annot):
    segs =  []
    for seg in annot['segmentation']:
        # print(seg, len(seg),  '\n')
        segs.append({
            'label': 'person',
            'points': parse_segmentation(seg),
            'group_id': int(annot['id']), 
            'shape_type': 'polygon',
            'flags': {}
        })
         
    box = {
        'label': 'person',
        'points': parse_bbox(annot['bbox']),
        'group_id': int(annot['id']),
        'shape_type': 'rectangle',
        'flags': {}
    }
    
    return segs, box


def get_image_id(annotation):
    image_id_person = []
    for annot in annotation:
        if annot['category_id'] == 1 and annot['iscrowd'] == 0:
            image_id_person.append(annot['image_id'])
            
    return image_id_person


# test_ids example = [101172, 117891, 120021, 143908, 184659, 209468, 390241, 403255, 425226, 537548]
test_ids = get_image_id(coco_instance_json['annotations'][:5000])

for i, image_id in enumerate(test_ids):
    boxs = []
    segs = []
    for annot in coco_instance_json['annotations']:    
        if annot['image_id'] == image_id:
            if annot['category_id'] == 1 and annot['iscrowd'] == 0:
                # pprint.pprint(annot)
                segs.append(get_annotation(annot)[0]) 
                boxs.append(get_annotation(annot)[1]) 
    
   
    for seg in segs:
        if len(seg) > 1:
            boxs += [s for s in seg]
        else:
            boxs.append(seg[0])

    filename = file_name(image_id)
    
    labelme = {
        'version': '4.2.10',
        'flags': {},
        'shapes': boxs,  
        'imagePath': filename + '.jpg',
        'imageData': None,
        'imageHeight': get_size(image_path+filename+'.jpg')[0],
        'imageWidth': get_size(image_path+filename+'.jpg')[1],
    }

    # Save json files
    with open(copy_path+filename+'.json', 'w') as f:
        json.dump(labelme, f, indent=2)
        print(str(i+1)+'th files created!'+': '+filename+'.json'+'||'+filename+'.jpg')
    

    # Save image files

    src_dir = image_path+filename+'.jpg'
    dst_dir = copy_path+filename+'.jpg'
    shutil.copy(src_dir,dst_dir)




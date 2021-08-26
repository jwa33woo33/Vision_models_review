import json
import os
from statistics import mean, median
from collections import defaultdict

import numpy as np
import pprint as pp


"""
COCO image
1. width height
frequency
(mean, max, min)

2. Number of objects per image

3. Number of points per object + image(max, min, mean)
"""

json_path_2014 = './COCO/annotations_trainval2014/annotations/'
json_path_2017 = './COCO/annotations_trainval2017/annotations/'

# Read coco format 2014
with open(json_path_2014+'instances_val2014.json', 'r') as f:
    coco_json_2014 = json.load(f)

# Read coco format 2014
with open(json_path_2017+'instances_val2017.json', 'r') as f:
    coco_json_2017 = json.load(f)


height_2014 = []
width_2014 = []
height_2017 = []
width_2017 = []

for imgs in coco_json_2014['images']:
    height_2014.append(imgs['height'])
    width_2014.append(imgs['width'])

for imgs in coco_json_2017['images']:
    height_2017.append(imgs['height'])
    width_2017.append(imgs['width'])

print('COCO dataset 2014 info')
print(f'Height mean: {mean(height_2014):.3f}')
print(f'Width mean: {mean(width_2014):.3f}')

print(f'Height max: {max(height_2014)}')
print(f'Width max: {max(width_2014)}')

print(f'Height min: {min(height_2014)}')
print(f'Width min: {min(width_2014)}')

print('\n')

print('COCO dataset 2017 info')
print(f'Height mean: {mean(height_2017):.3f}')
print(f'Width mean: {mean(width_2017):.3f}')

print(f'Height max: {max(height_2017)}')
print(f'Width max: {max(width_2017)}')

print(f'Height min: {min(height_2017)}')
print(f'Width min: {min(width_2017)}')

"""
EDA
"""

{'segmentation': [[567.98, 206.38, 571.35, 569.6, 207.05, 568.38, 206.65]], 
'area': 75.42420000000037, 
'iscrowd': 0, 
'image_id': 458750, 
'bbox': [567.98, 190.2, 9.44, 16.85], 
'category_id': 1, 
'id': 548581
}


# Get merged points for polygon
merged = defaultdict(list)

# Extract only group_id and points while merging points by group_id
for annot in coco_json_2014['annotations']:
    merged[annot['image_id']].extend(annot['segmentation'])

merged_dict = [{'image_id': key, 'segmentation': merged_list} for key, merged_list in merged.items()]

# Sort dictionary by group_id
merged_dict = sorted(merged_dict, key=lambda k: k['image_id']) 

# pp.pprint(merged_dict)

num_obj = []
num_points = []

for seg in merged_dict:
    seg_shape = np.array(seg['segmentation']).shape
    num_obj.append(seg_shape[0])
    flatten_points = [y for x in seg['segmentation'] for y in x]
    num_points.append(len(flatten_points)/2)

print('\n')

print('COCO dataset 2014 Number of objects per image')
print(f'Minimum number of objects: {min(num_obj)}')
print(f'Maximum number of objects: {max(num_obj)}')
print(f'Mean number of objects: {mean(num_obj):.3f}')
print(f'Q1: {np.percentile(num_obj, 25)}, Q2: {np.percentile(num_obj, 50)}, Q3: {np.percentile(num_obj, 75)}')


print('\n')

print('COCO dataset 2014 Number of points per image')
print(f'Minimum number of points: {int(min(num_points))}')
print(f'Maximum number of points: {int(max(num_points))}')
print(f'Median number of points: {int(median(num_points))}')
print(f'Mean number of points: {mean(num_points):.3f}')
print(f'Q1: {np.percentile(num_points, 25)}, Q2: {np.percentile(num_points, 50)}, Q3: {np.percentile(num_points, 75)}')


# Get merged points for polygon
merged = defaultdict(list)

# Extract only group_id and points while merging points by group_id
for annot in coco_json_2017['annotations']:
    merged[annot['image_id']].extend(annot['segmentation'])

merged_dict = [{'image_id': key, 'segmentation': merged_list} for key, merged_list in merged.items()]

# Sort dictionary by group_id
merged_dict = sorted(merged_dict, key=lambda k: k['image_id']) 


num_obj = []
num_points = []

for seg in merged_dict:
    seg_shape = np.array(seg['segmentation']).shape
    num_obj.append(seg_shape[0])
    flatten_points = [y for x in seg['segmentation'] for y in x]
    num_points.append(len(flatten_points)/2)

print('\n')

print('COCO dataset 2017 Number of objects per image')
print(f'Minimum number of objects: {min(num_obj)}')
print(f'Maximum number of objects: {max(num_obj)}')
print(f'Mean number of objects: {mean(num_obj):.3f}')
print(f'Q1: {np.percentile(num_obj, 25)}, Q2: {np.percentile(num_obj, 50)}, Q3: {np.percentile(num_obj, 75)}')

print('\n')

print('COCO dataset 2017 Number of points per image')
print(f'Minimum number of points: {int(min(num_points))}')
print(f'Maximum number of points: {int(max(num_points))}')
print(f'Median number of points: {int(median(num_points))}')
print(f'Mean number of points: {mean(num_points):.3f}')
print(f'Q1: {np.percentile(num_points, 25)}, Q2: {np.percentile(num_points, 50)}, Q3: {np.percentile(num_points, 75)}')


"""
COCO dataset 2014 info
Height mean: 485.070
Width mean: 576.541
Height max: 640
Width max: 640
Height min: 111
Width min: 120


COCO dataset 2017 info
Height mean: 483.543
Width mean: 573.755
Height max: 640
Width max: 640
Height min: 145
Width min: 200


COCO dataset 2014 Number of objects per image
Minimum number of objects: 1
Maximum number of objects: 84
Mean number of objects: 8.344
Q1: 2.0, Q2: 5.0, Q3: 11.0


COCO dataset 2014 Number of points per image
Minimum number of points: 4
Maximum number of points: 1454
Median number of points: 122
Mean number of points: 176.293
Q1: 63.0, Q2: 122.0, Q3: 234.0


COCO dataset 2017 Number of objects per image
Minimum number of objects: 1
Maximum number of objects: 71
Mean number of objects: 8.544
Q1: 2.0, Q2: 5.0, Q3: 12.0


COCO dataset 2017 Number of points per image
Minimum number of points: 4
Maximum number of points: 1263
Median number of points: 125
Mean number of points: 177.708
Q1: 64.0, Q2: 125.0, Q3: 233.0
"""
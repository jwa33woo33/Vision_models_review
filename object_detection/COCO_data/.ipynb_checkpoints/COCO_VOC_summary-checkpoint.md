# 1. COCO dataset

Website: http://cocodataset.org/

## COCO dataset is composed by .json format

### High level format is described as below


```python
{
    "info": {...},
    "licenses": [...],
    "images": [...],
    "annotations": [...],
    "categories": [...],
    "segment_info": [...]
}
```


#### 1) INFO

- Provides information summary of data. So when we create our own data, we can apply our own information to this.

```python
"info": {
    "description": "36 Research Dataset", # Dataset description
    "url": "http://36.edu", # URL
    "version": "1.0", # version
    "year": 2017, # year
    "contributor": "36 University", # group represents object in context
    "date_created": "2030/01/01" # date when data created
}
```


#### 2) LICENSES

- Legal documents. Terms and condition

```python
"licenses": [
    {
        "url": "http://36.edu/licenses/", # License URL
        "id": 1, # License ID
        "name": "36 Research License" # License Name
    },
    ... # and More.....
]
```

#### 3) IMAGES

- List of image, and shows information of image as below

```python
"images": [
    {
        "id": 3333 # image id
        "license": 1, #this license id is corresponding to license id in license section
        "file_name": "000000003333.jpg", # 화일 name. I guess usually be identical with its id when file name is organized by sorted order. Not necessarily be identical
        "coco_url": "http://36.edu/val2020/0000000003333.jpg", # 코코 URL
        "flickr_url": "http://farm9.staticflickr.com/8429/1234.jpg", # 플리커 URL
        "height": 333,
        "width": 500,
        "date_captured": "2030-01-01 03:33:33",
        
    },
    ... # and More.....
]
   
```
#### 4) ANNOTATIONS

- Annotation section contains actual data we can apply to model

```python
"annotations": [
    {
        "id": 1768 # Unique id of each segmentation
        "image_id": 289343, # Each image id
        "category_id": 18, # Category id that represent name of object: ex) ballon, person, bicycle and so on
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]], # Segmentation is list of coordinates(vertices) that defines the shape as many X,Y coordinate ex) [[x,y,x,y,x,y,x,y, ... x,y]] and each x,y, point represent its coodinate. Also data type is "list" of "list" because each shaded has a single bracket [] so if we have 2 chunks of shaded area in a single object, then it would be [[x,y...],[x,y...]
        "area": 702.1057499999998, # Number of pixels in shaded area
        "iscrowd": 0, # Provide an information that if shaded area are crowded or not       
        "bbox": [473.07,395.93,38.65,28.67], # [X, Y, width, height] and X, Y coordinates are starting from top left corner of image.
        
        
    },
    ...
    # Case of when iscrowd: 1
    {
        "segmentation": {
            "counts": [179,27,392,41,…,55,20], # counts number of pixcels of non-shaded area from top left corner, direction to the right, then goes to second line of pixel, then direction to the right until it reaches object exist and counts over and over again.
            "size": [426,640]
        },
        "area": 220834,
        "iscrowd": 1,
        "image_id": 250282,
        "bbox": [0,34,639,388],
        "category_id": 1,
        "id": 900100250282
    }
]
```
#### 5) CATEGORIES(This section is not in Captions annotations)

- This section provides category information. supercategory describes name of larger category(group), and name is name of subcategory

```python
"categories": [
    {"supercategory": "person",
     "id": 1,
     "name": "person"},
    {"supercategory": "vehicle",
     "id": 2,
     "name": "bicycle"},
    {"supercategory": "vehicle",
     "id": 3,
     "name": "car"},
    {"supercategory": "vehicle",
     "id": 4,
     "name": "motorcycle"},
    {"supercategory": "vehicle",
     "id": 5,
     "name": "airplane"},
    ...
    {"supercategory": "indoor",
     "id": 89,
     "name": "hair drier"},
    {"supercategory": "indoor",
     "id": 90,
     "name": "toothbrush"}
]
```

#### 6) SEGMENT_iNFO(This section is only in Panoptic annotations)

------------------------


## Tasks:
- There are 5 tasks that we can do with COCO dataset, and each task contain its own data format: 

<font color='red'>
Detection | Keypoints | Stuff | Panoptic | Captions
</font>

#### 1) Object detection Format

- The COCO Object Detection Task is designed to push the state of the art in object detection forward. COCO features two object detection tasks: using either bounding box output or object segmentation output (the latter is also known as instance segmentation).

![Image of Yaktocat](http://cocodataset.org/images/detection-splash.png)

Data Format:
```python
annotation{
    "id" : int, 
    "image_id" : int, 
    "category_id" : int, 
    "segmentation" : RLE or [polygon], 
    "area" : float, 
    "bbox" : [x,y,width,height], 
    "iscrowd" : 0 or 1,
}

categories[{
    "id" : int, 
    "name" : str, 
    "supercategory" : str,
}]
```


#### 2) Key Point Detection Format

- The COCO Keypoint Detection Task requires localization of person keypoints in challenging, uncontrolled conditions. The keypoint task involves simultaneously detecting people and localizing their keypoints (person locations are not given at test time).

![Image of Yaktocat](http://cocodataset.org/images/keypoints-splash.png)


Data Format:
```python
annotation{
    "keypoints" : [x1,y1,v1,...], 
    "num_keypoints" : int, 
    "[cloned]" : ...,
}

categories[{
    "keypoints" : [str], 
    "skeleton" : [edge], 
    "[cloned]" : ...,
}]

# "[cloned]": denotes fields copied from object detection annotations defined above.

```


#### 3) Stuff Segmentation Format

- The COCO Stuff Segmentation Task is designed to push the state of the art in semantic segmentation of stuff classes. Whereas the object detection task addresses thing classes (person, car, elephant), this task focuses on stuff classes (grass, wall, sky).


![Image of Yaktocat](http://cocodataset.org/images/stuff-splash.png)

Data Format:
```python
# Same as object detection data format except for "iscrowd"

```

#### 4) Panoptic Segmentation

- The COCO Panoptic Segmentation Task is designed to push the state of the art in scene segmentation. Panoptic segmentation addresses both stuff and thing classes, unifying the typically distinct semantic and instance segmentation tasks. The aim is to generate coherent scene segmentations that are rich and complete, an important step toward real-world vision systems such as in autonomous driving or augmented reality.

![Image of Yaktocat](http://cocodataset.org/images/panoptic-splash.png)

Data Format:
```python
annotation{
    "image_id" : int, 
    "file_name" : str, 
    "segments_info" : [segment_info],
}

segment_info{
    "id" : int,
    "category_id" : int, 
    "area" : int, 
    "bbox" : [x,y,width,height], 
    "iscrowd" : 0 or 1,
}

categories[{
    "id" : int, 
    "name" : str, 
    "supercategory" : str, 
    "isthing" : 0 or 1, 
    "color" : [R,G,B],
}]

```

#### 5) Image Captioning

- The COCO Captioning Challenge is designed to spur the development of algorithms producing image captions that are informative and accurate.


![Image of Yaktocat](http://cocodataset.org/images/captions-splash.jpg)

Data Format:
```python
annotation{
    "id" : int, 
    "image_id" : int, 
    "caption" : str,
}

```



# 2. PASCAL VOC dataset


Website: http://host.robots.ox.ac.uk/pascal/VOC/


## PASCAL VOC dataset is composed by .xml format

- Each image contains corresponding xml file and one of examples is shown as below.


```python
<annotation>
    <folder>VOC2007</folder>
    <filename>000001.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>341012865</flickrid>
    </source>
    <owner>
        <flickrid>Fried Camels</flickrid>
        <name>Jinky the Fruit Bat</name>
    </owner>
    <size>
        <width>353</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>dog</name>
        <pose>Left</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>48</xmin>
            <ymin>240</ymin>
            <xmax>195</xmax>
            <ymax>371</ymax>
        </bndbox>
    </object>
    <object>
        <name>person</name>
        <pose>Left</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>8</xmin>
            <ymin>12</ymin>
            <xmax>352</xmax>
            <ymax>498</ymax>
        </bndbox>
    </object>
</annotation>
```


- ```<folder>``` Name of Parent folder where containing both xml files, and images
- ```<filename>``` Name of the physical file that exists in the folder
- ```<size>``` : Tag of width, height, and channel information of an image
- ```<object>``` : Object information that contains class name, bndbox coordinates
- ```<name>``` : Class name
- ```<truncated>``` : Indicates that the bounding box specified for the object does not correspond to the full extent of the object. For example, if an object is visible partially in the image then we set truncated to 1. If the object is fully visible then set truncated to 0
- ```<difficult>``` : An object is marked as difficult when the object is considered difficult to recognize. If the object is difficult to recognize then we set difficult to 1 else set it to 0
- ```<bndbox> ```: Tag of xmin, ymin, xmax, and ymax
- ```<xmin>``` : X Coordinate. top left corner of bounding box
- ```<ymin>``` : Y Coordinate. top left corner of bounding box
- ```<xmax>``` : X Coordinate. bottom right corner of bounding box
- ```<ymax>``` : Y Coordinate. bottom right corner of bounding box
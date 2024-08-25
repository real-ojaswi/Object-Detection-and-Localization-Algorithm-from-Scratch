
from pycocotools.coco import COCO
import cv2
import json
import os
import torch
import random


# seed = 121
seed =455
torch.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Path to COCO dataset annotations
train_annotation_path = '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Coco/annotations/instances_train2017.json' #location of train annotation
val_annotation_path = '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Coco/annotations/instances_val2017.json' #location of val annotation
train_image_path= '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Coco/train2017' #location of train images
val_image_path= '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Coco/val2017' #location of val images


# Categories of interest
categories_of_interest = ['cake', 'dog', 'motorcycle'] #list of categories by name
min_object_area = 4096
target_image_size = (256, 256)


# Function to resize image and scale bounding box coordinates
def resize_and_scale_bbox(image, bbox, target_size):
    resized_image = cv2.resize(image, target_size)
    scale_x = target_size[0] / image.shape[1]
    scale_y = target_size[1] / image.shape[0]
    x, y, w, h = bbox
    resized_bbox = [int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)]
    return resized_image, resized_bbox


# Load COCO dataset
coco_train = COCO(train_annotation_path)
coco_val = COCO(val_annotation_path)


def get_category_id(category_name):
    cat_ids = coco_train.getCatIds(catNms=[category_name])
    if cat_ids:
        return cat_ids[0]
    else:
        return None
category_ids_of_interest= [get_category_id(category) for category in categories_of_interest]
category_id_to_class_map= {key:value for (key, value) in zip(category_ids_of_interest, categories_of_interest) }
category_id_to_classid_map= {key:value for (key, value) in zip(category_ids_of_interest, [0, 1, 2]) }


# Filter and process images for training set
train_data = []
for img_id in coco_train.imgs:
    ann_ids = coco_train.getAnnIds(imgIds=img_id)
    anns = coco_train.loadAnns(ann_ids)
    for ann in anns:
        if ann['category_id'] in category_ids_of_interest and ann['area'] > min_object_area:
            image_path = os.path.join(train_image_path, coco_train.imgs[img_id]['file_name'])
            image = cv2.imread(image_path)
            resized_image, resized_bbox = resize_and_scale_bbox(image, ann['bbox'], target_image_size)
            train_data.append({
                'image_name': coco_train.imgs[img_id]['file_name'],
                'image_size': target_image_size,
                'resized_image': resized_image,  # Add resized image
                'bbox': resized_bbox,
                'category_id': ann['category_id'],
                'class_name': category_id_to_class_map[ann['category_id']], #using the mapping to save the class name
                'class_id' : category_id_to_classid_map[ann['category_id']], #using the mapping to save the class id
                'segmentation': ann['segmentation']
            })

# Filter and process images for testing set
test_data = []
for img_id in coco_val.imgs:
    ann_ids = coco_val.getAnnIds(imgIds=img_id)
    anns = coco_val.loadAnns(ann_ids)
    for ann in anns:
        if ann['category_id'] in category_ids_of_interest and ann['area'] > min_object_area:
            image_path = os.path.join(val_image_path, coco_val.imgs[img_id]['file_name'])
            image = cv2.imread(image_path)
            resized_image, resized_bbox = resize_and_scale_bbox(image, ann['bbox'], target_image_size)
            test_data.append({
                'image_name': coco_val.imgs[img_id]['file_name'],
                'image_size': target_image_size,
                'resized_image': resized_image,  # Add resized image
                'bbox': resized_bbox,
                'category_id': ann['category_id'],
                'class_name': category_id_to_class_map[ann['category_id']], #using the mapping to save the class name
                'class_id' : category_id_to_classid_map[ann['category_id']], #using the mapping to save the class id
                'segmentation': ann['segmentation']
            })

# Save the processed images and annotations to disk for both training and testing sets
train_output_dir= '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Extracted_CocoLOAD/train' #location of extracted training images
test_output_dir= '/local/scratch/a/oachary/Classes/Semester2/ECE60146/Datasets/Extracted_CocoLOAD/validation' #location of extracted validation images


# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Save training images and annotations
saved_image_train= [] #to make sure that the same image is not repeated
for i, data in enumerate(train_data):
    if data['image_name'] in saved_image_train:
        continue
    saved_image_train.append(data['image_name'])
    image_name = f"train_{i}.jpg"
    cv2.imwrite(os.path.join(train_output_dir, image_name), data['resized_image'])
    annotations = []
    for annotation_data in train_data:
        if annotation_data['image_name'] == data['image_name']:
            annotations.append({
                'bbox': annotation_data['bbox'],
                'category_id': annotation_data['category_id'],
                'class_name': annotation_data['class_name'],
                'class_id': annotation_data['class_id'],
                'segmentation': annotation_data['segmentation']
            })
    category_ids = [annotation['category_id'] for annotation in annotations]
    class_names= [annotation['class_name'] for annotation in annotations]
    class_ids= [annotation['class_id'] for annotation in annotations]
    bboxes= [annotation['bbox'] for annotation in annotations]
    segmentations= [annotation['segmentation'] for annotation in annotations]
    annotation_filename = f"train_{i}.json"
    with open(os.path.join(train_output_dir, annotation_filename), 'w') as f:
        json.dump({
            'image_name': image_name,
            'image_size': target_image_size,
            'category_ids': category_ids,
            'class_names': class_names,
            'class_ids': class_ids,
            'bboxes': bboxes,
            'segmentations': segmentations,
            'num_objects': len(category_ids),
        }, f)
        

# Save testing images and annotations
saved_image_test= [] #to make sure that the same image is not repeated
for i, data in enumerate(test_data):
    if data['image_name'] in saved_image_test:
        continue
    saved_image_test.append(data['image_name'])
    image_name = f"test_{i}.jpg"
    cv2.imwrite(os.path.join(test_output_dir, image_name), data['resized_image'])
    annotations = []
    for annotation_data in test_data:
        if annotation_data['image_name'] == data['image_name']:
            annotations.append({
                'bbox': annotation_data['bbox'],
                'category_id': annotation_data['category_id'],
                'class_name': annotation_data['class_name'],
                'class_id': annotation_data['class_id'],
                'segmentation': annotation_data['segmentation']
            })
    category_ids = [annotation['category_id'] for annotation in annotations] #for multi-object images, 
    class_names= [annotation['class_name'] for annotation in annotations]    #saving properties of several objects as a list
    class_ids= [annotation['class_id'] for annotation in annotations]
    bboxes= [annotation['bbox'] for annotation in annotations]
    segmentations= [annotation['segmentation'] for annotation in annotations]
    annotation_filename = f"test_{i}.json"
    with open(os.path.join(test_output_dir, annotation_filename), 'w') as f: 
        json.dump({                                                           #dumping the files into a json file
            'image_name': image_name,
            # 'image_size': target_image_size,
            'category_ids': category_ids,
            'class_names': class_names,
            'class_ids': class_ids,
            'bboxes': bboxes,
            'segmentations': segmentations,
            'num_objects': len(category_ids),
        }, f)



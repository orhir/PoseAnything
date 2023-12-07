import json
import os
import shutil
import sys
import numpy as np
from xtcocotools.coco import COCO


def search_match(bbox, num_keypoints, segmentation):
    found = []
    checked = 0
    for json_file, coco in COCO_DICT.items():
        cat_ids = coco.getCatIds()
        for cat_id in cat_ids:
            img_ids = coco.getImgIds(catIds=cat_id)
            for img_id in img_ids:
                annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_id))
                for ann in annotations:
                    checked += 1
                    if (ann['num_keypoints'] == num_keypoints and ann['bbox'] == bbox and ann[
                        'segmentation'] == segmentation):
                        src_file = coco.loadImgs(img_id)[0]["file_name"]
                        split = "test" if "test" in json_file else "train"
                        found.append((src_file, ann, split))
                        # return src_file, ann, split
    if len(found) == 0:
        raise Exception("No match found out of {} images".format(checked))
    elif len(found) > 1:
        raise Exception("More than one match! ".format(found))
    return found[0]

if __name__ == "__main__":

    carfusion_dir_path = sys.argv[1]
    mp100_dataset_path = sys.argv[2]
    os.makedirs('output', exist_ok=True)
    for cat in ['car', 'bus', 'suv']:
        os.makedirs(os.path.join('output', cat), exist_ok=True)


    COCO_DICT = {}
    ann_files = os.path.join(carfusion_dir_path, 'annotations')
    for json_file in os.listdir(ann_files):
        COCO_DICT[json_file] = COCO(os.path.join(carfusion_dir_path, 'annotations', json_file))

    count = 0
    print_log = []
    for json_file in os.listdir(mp100_dataset_path):
        print("Processing {}".format(json_file))
        cats = {}
        coco = COCO(os.path.join(mp100_dataset_path, json_file))
        cat_ids = coco.getCatIds()
        for cat_id in cat_ids:
            category_info = coco.loadCats(cat_id)
            cat_name = category_info[0]['name']
            if cat_name in ['car', 'bus', 'suv']:
                cats[cat_name] = cat_id


        for cat_name, cat_id in cats.items():
            img_ids = coco.getImgIds(catIds=cat_id)
            count += len(img_ids)
            print_log.append(f'{json_file} : {cat_name}: {len(img_ids)}')
            for img_id in img_ids:
                img = coco.loadImgs(img_id)[0]
                dst_file_name = img['file_name']
                annotation = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=None))
                bbox = annotation[0]['bbox']
                keypoints = annotation[0]['keypoints']
                segmentation = annotation[0]['segmentation']
                num_keypoints = annotation[0]['num_keypoints']

                # Search for a match:
                src_img, src_ann, split = search_match(bbox, num_keypoints, segmentation)
                shutil.copyfile(
                    os.path.join(carfusion_dir_path, split, src_img),
                    os.path.join('output', dst_file_name))
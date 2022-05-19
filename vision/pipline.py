"""
Takes information from the camera and returns interop output json file.
"""

#TODO: Make sure to pass focal length metadata for de-skew
# pull from other branch: git pull origin feature/stitching

import os
import sys
import json
import concurrent.futures as futures
import time
from xmlrpc.client import Boolean
import cv2
import numpy as np
from PIL import Image
import numpy.typing as npt
from typing import List, Tuple

parent_dir = os.path.dirname(os.path.abspath(__file__))

# Images for stitching
stitch_dir = parent_dir + "\mapping\images"

# Images for odlc
image_dir = parent_dir + "\images"

from mapping.image_stitcher import Stitcher

def get_focal(img_jpg: npt.NDArray[np.uint8]) -> float:
    image = Image.open(img_jpg)
    exifdata = image._getexif()
    focal_length = exifdata.get(37386)

    # for tag_id in exifdata:
    #     tag = TAGS.get(tag_id,tag_id)
    #     data = exifdata.get(tag_id)
    #     if isinstance(data,bytes):
    #         data=data.decode()
    #     print(f"{tag:20}:{data}", " ", tag_id)

    return focal_length

def obj_in_frame(image: npt.NDArray[np.uint8], index: int, focal_length: float) -> Tuple(int, npt.NDArray[np.uint8]):
    time.sleep(4)
    print("Checking obj in frame...")
    # Returns list of bounding boxes and index image.
    
    return index, [1]

def raw_img_process(r_image, b_boxes):
    print("RAW image Processing")
    time.sleep(2)
    return

def img_scheduler(stitch: Stitcher, list_jpg: List[npt.NDArray[np.uint8]], list_raw: List[npt.NDArray[np.uint8]], list_focal: List[float]):
    executor = futures.ThreadPoolExecutor()
    check_frame = [executor.submit(obj_in_frame, image, index, list_focal[index]) for index, image in enumerate(list_jpg)]
    # img_stitch =  executor.submit(stitch.multiple_image_stitch())

    # while True:
    #     if(img_stitch.done()):
    #         print("Stitch Done")
    #         break

    for detection in futures.as_completed(check_frame):
        index, b_boxes = detection.result()
        if b_boxes != None:
            img_process = executor.submit(raw_img_process, list_raw[index], b_boxes)

    return img_process

def init_vision():
    stitch = Stitcher()
    stitch.image_path = stitch_dir

    list_jpg = []
    list_raw = []
    list_focal = []

    if not os.listdir(stitch_dir):
        raise IndexError("No Images")

    for file in os.listdir(stitch_dir):
        img = os.path.join(stitch_dir, file)
        if file.endswith(".JPG"):
            list_focal.append(get_focal(img))
            list_jpg.append(cv2.imread(img))
        if file.endswith(".RAW"):
            list_raw.append(cv2.imread(img))

    start = time.time()
    res = img_scheduler(stitch, list_jpg, list_raw, list_focal)
    while True:
        if res.done():
            end = time.time()
            print("MULTIPROC", end-start)
            break

if __name__ == '__main__':
    init_vision()
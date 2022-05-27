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
import logging

from mapping.image_stitcher import Stitcher

parent_dir = os.path.dirname(os.path.abspath(__file__))

class Pipeline:
    def __init__(self) -> None:
        self.compute_images = []
        self.stitch_images = []
        self.stitch_dir = parent_dir + "\mapping\images"
        self.image_dir = parent_dir + "\images"
        self.mapping_dir = parent_dir + "\mapping"

    def get_focal(self, img_jpg: npt.NDArray[np.uint8]) -> float:
        image = Image.open(img_jpg)
        exifdata = image._getexif()
        focal_length = exifdata.get(37386)
        return focal_length

    def obj_in_frame(self, name: str, image: npt.NDArray[np.uint8], focal_length: float):
        time.sleep(4)
        print("Checking if " +  name + " in frame...")
        # Returns list of bounding boxes and index image.
        
        return name, [1]

    def raw_img_process(self, image, b_boxes):
        print("RAW image Processing")
        time.sleep(2)
        return

    def img_scheduler(self, stitch, list_image):
        executor = futures.ProcessPoolExecutor()
        check_frame = [executor.submit(self.obj_in_frame, image[0], cv2.imread(image[0] + ".JPG"), list_image[1]) for  image in list_image]

        for detection in futures.as_completed(check_frame):
            name, b_boxes = detection.result()
            if b_boxes != None:
                img_process = executor.submit(self.raw_img_process, cv2.imread(name + ".RAW"), b_boxes)

        img_stitch = stitch.multiple_image_stitch()
        cv2.imwrite(os.path.join(self.mapping_dir , 'FINAL.JPG'), img_stitch)
        print("Stitch Done")

        return img_process

    def init_vision(self):
        stitch = Stitcher()
        stitch.image_path = self.stitch_images
        list_image = []

        for file in self.compute_images:
            if file.endswith(".JPG"):
                img = os.path.join(self.stitch_dir, file)
                list_image.append([file[:-4], self.get_focal(img)])

        res = self.img_scheduler(stitch, list_image)

if __name__ == '__main__':
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # logger config
    file = logging.FileHandler(filename = "logger.log", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)

    # logger.debug("WHAT")

    pipe = Pipeline()

    stitch_dir = parent_dir + "\mapping\images"

    os.chdir(stitch_dir)

    for file in os.listdir(stitch_dir):
        pipe.compute_images.append(file)
        pipe.stitch_images.append(file)

    pipe.init_vision()
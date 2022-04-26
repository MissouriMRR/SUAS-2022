"""
Takes information from the camera and returns interop output json file.
"""

import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, wait
import time
import cv2
import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))
stitch_dir = parent_dir + "\mapping"
image_dir = parent_dir + "\images"

from mapping.image_stitcher import Stitcher

def run_modules(image):
    print("start task")
    time.sleep(2)
    print("doing task")

def img_processor(stitch, image_list):
    with ProcessPoolExecutor() as executor:
        img_proccess = [executor.submit(run_modules, image) for image in image_list]
        img_stitch =  executor.submit(stitch.multiple_image_stitch())
        wait(img_proccess)
        while True:
            if(img_stitch.done()):
                print("All done")
                break

if __name__ == '__main__':
    stitch = Stitcher()
    stitch.image_path = stitch_dir

    image_list = []

    if not os.listdir(stitch_dir):
        raise IndexError("No Images")

    for file in os.listdir(stitch_dir):
        if file.endswith(".jpg"):
            image_list.append(cv2.imread(os.path.join(stitch_dir, file)))

    start = time.time()
    img_processor(stitch, image_list)
    end = time.time()
    print("MULTIPROC", end-start)

    image = np.array([])
    start = time.time()
    for _ in range(4):
        run_modules(image)
        stitch.multiple_image_stitch()
    end = time.time()
    print("NORMAL", end-start)
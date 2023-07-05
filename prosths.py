""" 
Project Author: Chukwuemeka L. Nkama
Date: July 5, 2023

prosths.py is a python file that contains an helper class
and functions for object detection and segmentation of human 
body parts...
"""

# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np 


class Prosth():
    def __init__(self, image_path):
        """Initialize YOLO architecture
           for detecting and segmenting
           objects in an image.

           Arguments
           ----------
           image_path: path to image
        """
        self.__model = YOLO("yolov8m-seg.pt")
        self.__img = cv2.imread(image_path)
        self.__height, self.__width = self.__img.shape

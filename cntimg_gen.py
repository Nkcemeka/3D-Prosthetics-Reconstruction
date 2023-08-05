"""
Project Author: Chukwuemeka L. Nkama
Date: July 31, 2023

This file generates an image containing the
contour of interest!
"""
import sys
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

if len(sys.argv) == 3:
    img_path = sys.argv[1]
    file_path = sys.argv[2]
else:
    print("ERROR! Format is python3 cntimg_gen.py img_path file_path")
    sys.exit()

# Read Image
img = cv2.imread(img_path)

# Get angle of rotation
with open('rotinfo.txt', 'r') as filereader:
    rot_angle = filereader.read()

# Remove trailing and leading whitespace characters
rot_angle = rot_angle.strip()

# Get the angle
if 'n' in rot_angle:
    rot_angle = rot_angle.replace("n", '')
elif 'acw' in rot_angle:
    rot_angle = rot_angle.replace('acw', '')
    rot_angle = '-'+rot_angle
else:
    rot_angle = rot_angle.replace('cw', '')

# convert angle to float
rot_angle = float(rot_angle)

# Rotate the image
rotMat = cv2.getRotationMatrix2D((0,0), np.degrees(rot_angle), 1)
img = cv2.warpAffine(img, rotMat, (img.shape[1], img.shape[0]))

# Read Contour file
cnts = pd.read_csv(".contours.csv")
cnts = cnts.iloc[:,:].to_numpy()
cnts = np.intp(cnts)
min_x = np.min(cnts[:,0]) # get min. x
max_x = np.max(cnts[:,0]) # get max. x
min_y = np.min(cnts[:,1]) # get min. y
max_y = np.max(cnts[:,1]) # get max. y

# Get the suitable x coords and y coords
x_suit = (np.arange(img.shape[1]) >= min_x) & (np.arange(img.shape[1]) <= max_x)
y_suit = (np.arange(img.shape[0]) >= min_y) & (np.arange(img.shape[0]) <= max_y)

x_coords = np.arange(img.shape[1])[x_suit]
y_coords = np.arange(img.shape[0])[y_suit]
x,y = np.meshgrid(x_coords, y_coords)
coords = np.column_stack((x.ravel(), y.ravel()))

# Get polygon
poly = Polygon(cnts.tolist())

# Get the final image
final_img = np.zeros(img.shape)

for col, row in coords:
    if not poly.contains(Point(col, row)):
        continue
    final_img[row, col] = img[row, col]

final_img.astype(np.uint8)
cv2.imwrite("cnt_image.png", final_img)

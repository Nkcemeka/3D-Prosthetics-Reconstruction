"""
Project Author: Chukwuemeka L. Nkama
Date: July 5, 2023

prosths.py is a python file that contains helper classes
and methods for object detection and segmentation of human
body parts...
"""

# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np
import sys
import pandas as pd

class Dseg():

    """
        Sets up deep learning model to
        detect and segment human body parts.
    """

    def __init__(self, image_path):
        """
            Initialize YOLO architecture
            for detecting and segmenting
            objects in an image.

            Arguments
            ----------
            image_path: path to image
        """
        self._img = cv2.imread(image_path)
        self._pred = None
        self._detect_flag = False 
        self._obj_id = 41  # object id in model
        self._contours = []
        self._prob = None 
        self._initRotFlag = False

    def detect(self, image):
        """
            Detects objects in image based on
            YOLO
        """
        if self._img is None:
            print("Image could not be read!")
            sys.exit()

        model = YOLO("yolov8m-seg.pt")
        preds = model.predict(image.copy())
        self._pred = preds[0]  # index 0th term since we have 1 img
        self._detect_flag = True

    def segment(self):
        """
            Finds the contours of detected
            objects in an image. To call this
            method, call detect() first.
        """

        if not self._detect_flag:
            print("ERROR: Call detect() first")
            sys.exit()

        if len(self._pred.boxes) == 0:
            print("No contour found!")
            sys.exit() 

        counter = 0  # helps skip contours not of interest
        for each_segment in self._pred.masks.xyn:
            if self._pred.boxes[counter].cls[0].item() != self._obj_id:
                counter += 1
                continue  # skip the contour

            # scale the segment
            each_segment[:, 0] *= self._img.shape[1]
            each_segment[:, 1] *= self._img.shape[0]
            contour = np.array(each_segment, dtype=np.intp)
            self._contours.append(contour)

            # Take note of probability
            self._prob = self._pred.boxes[counter].conf[0].item()
            counter += 1
        
        if len(self._contours) == 0:
            print("Objects found by model but not what you want!")
            sys.exit()
        
        self.get_min_rectBox() # Get min. rect that surrounds the image


class MinRect(Dseg):
    """
        This class inherits Dseg and attempts
        to obtain a Minimum Rectangle enclosing
        the contour of interest. This rectangle
        is to be drawn on one contour as we assume
        that the object of interest occurs once in
        the image. This class also gets the rotated
        contours and the coordinate frame of ref.
    """

    def __init__(self, image_path):
        super().__init__(image_path)
        self._rotFlag = None  # checks if obj is rotated
        self._minRectBox = None
        self._uvecs = None
        self._rotAngle = None 
        self._width = None
        self._height = None 
        self._center = None 
        self._crot = None 
        self._leftCnt = None 
        self._rightCnt = None 

    def get_min_rectBox(self):
        """
            Gets the box coordinates
            of the minimum area rect.
        """
        rect = cv2.minAreaRect(self._contours[0])  # select first object
        self._center = rect[0] # center coordinates of rectangle 

        if np.absolute(rect[2]) == 90:
            self._rotFlag = False
        else:
            self._rotFlag = True

        # Get the coords of the min. Area rect 
        rect_box = cv2.boxPoints(rect)
        self._minRectBox = np.intp(rect_box)

        # Get width and height of rectangle
        box = self._minRectBox.copy()
        box = box[box[:,1].argsort()]  # sort the box array

        # Get the unit vectors
        self.uvec(self._minRectBox)

    def uvec(self, box):
        """
            Uvec takes the min. rectangle box,
            sorts it to get the two highest points
            (top left and top right) at the top of
            the array. It then gets the unit vectors
            based on the minRect.

            Note that the top left corner is always
            selected to be the origin!

            Arguments
            ---------
            box: self._minRectBox
        """

        box = box.copy()
        box = box[box[:,1].argsort()]  # sort the box array
        
        # Get the width of the rectangle 
        self._width = np.linalg.norm(box[1] - box[0])

        # Get unit vectors for min. rect
        if box[1][1] != box[0][1]:
            if box[1][0] < box[0][0]: 
                self._crot = False  # rectangle was not rotated clockwise 
                uvec1 = box[0] - box[1]
                uvec2 = box[3] - box[1]
                self._height = np.linalg.norm(uvec2)
            else:
                self._crot = True # rectangle was rotated clockwise 
                uvec1 = box[1] - box[0]
                uvec2 = box[2] - box[0]
                self._height = np.linalg.norm(uvec2)
        else:
            if box[1][0] > box[0][0]:
                # box0 is the origin
                uvec1 = box[1] - box[0]
                if box[3][0] > box[2][0]:
                    # box 2 is on the side of box0 
                    uvec2 = box[2] - box[0] 
                    self._height = np.linalg.norm(uvec2)
                else:
                    # box3 is on the side of box0
                    uvec2 = box[3] - box[0]
                    self._height = np.linalg.norm(uvec2)
            else:
                # box1 is the origin 
                uvec1 = box[0] - box[1] 
                if box[3][0] > box[2][0]:
                    # box2 is on the side of box1 
                    uvec2 = box[2] - box[1]
                    self._height = np.linalg.norm(uvec2)
                else:
                    # box3 is on the side of box1 
                    uvec2 = box[3] - box[1]
                    self._height = np.linalg.norm(uvec2)

        # Get the first unit vector
        uvec1 = uvec1 / np.linalg.norm(uvec1)

        # Get the second unit vector
        uvec2 = uvec2 / np.linalg.norm(uvec2)

        self._uvecs = (uvec1, uvec2)
        scalar_proj = np.dot(np.array([1,0]), uvec1)
        self._rotAngle = np.arccos(scalar_proj) 


        if self._initRotFlag:
            # Avoid unnecesary printouts to console
            return

        # print rectangle's angle of rotation
        if self._crot is None:
            print(f'Rect\'s angle of rotation is: 0 degrees')
            rot = '0n'
        elif self._crot:
            print(f'Rect\'s angle of rotation is: {self._rotAngle} degrees clockwise')
            rot = f'{self._rotAngle}cw'
        else:
            print(f'Rect\'s angle of rotation is: {self._rotAngle} anticlockwise degrees')        
            rot = f'{self._rotAngle}acw'

        self._initRotFlag = True # Means initial angle of rot. has been determined
        self.gen_rot_info(rot)

    def gen_rot_info(self, rot):
        """
            Stores the rotation info
            in a file.
        """
        with open("rotinfo.txt", 'w') as filewriter:
            filewriter.write(f'{rot}')

    def get_rot_img(self):
        """
            Get the rotated image 
            if it exists and its 
            contour.
        """

        if not self._rotFlag:
            # if image is not rotated, skip this function
            return 
    
        image = self._img.copy()

        # Get final rotated image if the minimum rectangle is rotated
        if self._crot:
            transMat = cv2.getRotationMatrix2D((0,0), np.degrees(self._rotAngle), 1)
        else:
            transMat = cv2.getRotationMatrix2D((0,0), np.degrees(-self._rotAngle), 1)

        self._img = cv2.warpAffine(image, transMat,\
                (image.shape[1], image.shape[0]))

        # detect object 
        self.detect(self._img)

        # Get contours or segments  
        self._contours = [] # reinitialize the contours variable
        self.segment()

    def get_contour_halves(self):
        """
            Used to get contour get contour 
            halves after a contour has been 
            found!
        """
        x_thresh = self._center[0]
        left_half = self._contours[0][self._contours[0][:,0] < x_thresh]
        right_half = self._contours[0][self._contours[0][:,0] >= x_thresh]

        # sort the left and right contours based on y value
        self._leftCnt = left_half[left_half[:,1].argsort()]
        self._rightCnt = right_half[right_half[:,1].argsort()]

class WidthVar(MinRect):
    """
        This class calculates the width
        along the height of the min.
        area rectangle
    """
    def __init__(self, image_path):
        super().__init__(image_path)
        self._suitable_coords = None
        self._suitable_line = []
        self._varwidth = []
    
    def get_var_width(self):
        prev = 0
        sep = 25
        for cnt in self._leftCnt:
            yval = cnt[1] 

            if yval-prev < sep:
                continue

            prev = yval

            loc = np.searchsorted(self._rightCnt[:,1], yval)
            if self._rightCnt[:,1][loc] == yval:
                xmin = cnt[0]
                xmax = self._rightCnt[:, 0][loc]
                line = np.array([[xmin, xmax],[yval, yval]])
                width = xmax - xmin
                self._varwidth.append(width)
                self._suitable_line.append(line)
            else:
                xmin = cnt[0]
                xmax = self.lin_interp(loc, yval)
                line = np.array([[xmin, xmax], [yval, yval]])
                width = xmax - xmin
                self._varwidth.append(width)
                self._suitable_line.append(line)

    def lin_interp(self, loc, yn):
        """
            Performs a linear interpolation
            to get a missing x value due to
            effects of integer approx.
        """
        if (loc-1) != 0:
            x1 = self._rightCnt[:, 0][loc-1]
            x2 = self._rightCnt[:, 0][loc]
            y1 = self._rightCnt[:, 1][loc-1]
            y2 = self._rightCnt[:,1][loc]
        else:
            x2 = self._rightCnt[:, 0][loc]
            y2 = self._rightCnt[:, 1][loc]
            x1 = x2 
            y1 = y2 - 1 

        if x1 == x2:
            m = (y2-y1) # to prevent div. by zero
        else:
            m = (y2-y1)/(x2 - x1)
        xn = x1 + (yn - y1)/m
        return xn

class Prosths(WidthVar):
    """
        This class draws the object contour, 
        bounding box and the varying width.
    """

    def __init__(self, image_path, flip_contour):
        super().__init__(image_path)
        self._flip_cnt = flip_contour
        self._line_heights = {}

    def run(self):
        """
            This method sets up the entire
            detection of human body part,
            segmentation, measurements etc..
        """
        self.detect(self._img) # detect object
        self.segment() # segment object 
        self.get_rot_img() # get rotated image if min. Rect is rotated 
        self.get_contour_halves()
        self.get_var_width()
        self.show()
        self.save_contours()

    def draw_minrect(self):
        """
            Draws the min. area rectangle
            on the object to show it has been
            detected
        """
        cv2.polylines(self._img, [self._minRectBox], True, (255, 0, 0), 4)

    def draw_misc(self):
        """
            Draws miscellaneous details
            helpful to the user
        """
        # Get text to write on image
        text_str = f'Cup: {round(self._prob, 2)}'
        text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_PLAIN, 1, 2)

        # Height and width of rectangle to write text
        height_text = text_size[1] + 40 
        width_text = text_size[0] + 100

        # Get colour to use
        clr = (255, 0, 0)

        # coords of top left corner of rectangle 
        x1 = np.intp(self._center[0] - 0.5 * self._width) 
        y1 = np.intp(self._center[1] - 0.5 * self._height) 

        cv2.rectangle(self._img, (x1, y1-height_text), (x1+width_text,y1), \
                        clr, -1)
        cv2.putText(self._img, text_str, (x1+5, y1-20), cv2.FONT_HERSHEY_PLAIN, 2, \
                    (255, 255, 255), 2)


    def draw_uvec(self):
        """
            Draws the unit vectors on
            the minimum area rect.
        """
        sorted_box = self._minRectBox[self._minRectBox[:,1].argsort()]

        # Draw uvecs
        if sorted_box[1][0] < sorted_box[0][0]:  # box rotated left
            cv2.arrowedLine(self._img, np.intp(sorted_box[1]), \
                    np.intp(sorted_box[1]+self._uvecs[0]*50),  \
                    (0, 255, 0), 2, tipLength = 0.03)  # new x axis

            cv2.arrowedLine(self._img, np.intp(sorted_box[1]), \
                    np.intp(sorted_box[1]+self._uvecs[1]*50),   \
                    (0, 255, 0), 2, tipLength = 0.03)  # new y axis
        else: # box rotated right
            cv2.arrowedLine(self._img, np.intp(sorted_box[0]), \
                    np.intp(sorted_box[0]+self._uvecs[0]*50),  \
                    (0, 255, 0), 2, tipLength = 0.03)  # new x axis

            cv2.arrowedLine(self._img, np.intp(sorted_box[0]), \
                    np.intp(sorted_box[0]+self._uvecs[1]*50),  \
                    (0, 255, 0), 2, tipLength = 0.03)  # new y axis


    def draw_contours(self):
        contour = self._contours[0]
        cv2.polylines(self._img, [contour], True, (0, 0, 255), 4)


    def draw_width(self):
        id = 1 # Id is used to know the identity of the line we measure
        id_list = list(range(1, len(self._suitable_line)+1))
        y_list = [] # add y1 values since we didn't linearly interpolate the left half
        for count in range(len(self._suitable_line)):
            # get coords in standard basis 
            x1, x2 = self._suitable_line[count][0].astype(np.intp)
            y1, y2 = self._suitable_line[count][1].astype(np.intp)

            # Draw  arrowed lines (tiplength determines arrow head's length) 
            cv2.arrowedLine(self._img, (x1, y1), (x2, y2), (50, 127, 205), 2,\
                            tipLength = 0.03) 
            cv2.arrowedLine(self._img,(x2, y2), (x1, y1),(50, 127, 205), 2,\
                            tipLength = 0.03)

            # Get width and create text 
            width = self._varwidth[count]
            text_str = f'Width ({id}): {round(width, 2)}'
            text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_PLAIN, 1, 2)
            id += 1

            # Add y1 vals
            y_list.append(y1)

            # Get x and y pos for text 
            x_pos = int(0.5*(x1+x2-text_size[0]))
            y_pos = int(y1+20)

            # Place text underneath arrowed line
            cv2.putText(self._img, text_str, (x_pos, y_pos),\
                         cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

        # Add id and y values to _line_heights
        self._line_heights['id'] = id_list
        self._line_heights['y values'] = y_list
        line_height_frame = pd.DataFrame(self._line_heights)

        # save line_height_frame to csv file
        line_height_frame.to_csv('line_height.csv', index=False)

    def show(self):
        """
            Shows the image with the detection,
            contour and measurements
        """
        self.draw_minrect()  # draw the minimum rectangle
        self.draw_misc() # draw misc info. 
        self.draw_contours() # draw contour of desired obj
        self.draw_uvec() # draw the unit vectors
        self.draw_width() # draw the varying width of obj 

        # show the images 
        cv2.namedWindow("Prosths", cv2.WINDOW_NORMAL)
        cv2.imshow('Prosths', self._img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save the file for the CAD user to get necessary info w/out
        # running the program again
        cv2.imwrite('final_img.jpg',self._img)

    def save_contours(self):
        contours = self._contours[0]
        x_coord = contours[:,0]

        # Flip contour if it is not rightside up in 
        # CAD software
        if self._flip_cnt:
            y_coord = -1*contours[:,1]
        else:
            y_coord = contours[:,1]
        
        # contours[:,1] is to prevent unnecessary errors in other
        # code files.
        contour_frame = pd.DataFrame({'x':x_coord, 'y':contours[:,1]}) 
        contour_frame.to_csv(".contours.csv",index=False)
        

# Read Cmd Arguments
img_path = sys.argv[1] # image path
flip_cnt = int(sys.argv[2]) # Determines if contour should be flipped 

# Create Prosthetics object and run it
Prosths_obj = Prosths(img_path, flip_cnt)
Prosths_obj.run()

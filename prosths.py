"""
Project Author: Chukwuemeka L. Nkama
Date: July 5, 2021

prosths2.py is a python file that contains helper classes
and methods for object detection and segmentation of human
body parts...
"""

# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np

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
        self._segment_flag = False 
        self._obj_id = 41  # object id in model
        self._contours = []
        self._prob = None 

    def detect(self, image):
        """
            Detects objects in image based on
            YOLO
        """
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
            return

        if len(self._pred.boxes) == 0:
            print("No contour found!")
            return 

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
        
        self._segment_flag = True # contours were discovered
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
        self._width = np.linalg.norm(box[1] - box[0])
        self._height = np.linalg.norm(box[3] - box[0])

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
        flip = False  # used to set origin as top left corner

        # Get unit vectors for min. rect
        if box[1][0] < box[0][0]:  # rect is rot. left
            box = -1 * box  # mult. by -1 to get uvec in right dirxns
            flip = True
            self._crot = False  # rectangle was not rotated clockwise 
        else:
            self._crot = True # rectangle was rotated clockwise 

        # Get the first unit vector
        uvec1 = box[1] - box[0]
        uvec1 = uvec1 / np.linalg.norm(uvec1)

        # Get the second unit vector
        if flip:
            uvec2 = -box[3] + box[1]  # done since box was mult. by -1
            uvec2 = uvec2 / np.linalg.norm(uvec2)
        else:
            uvec2 = box[3] - box[0]
            uvec2 = uvec2 / np.linalg.norm(uvec2)

        self._uvecs = (uvec1, uvec2)
        scalar_proj = np.dot(np.array([1,0]), uvec1)
        self._rotAngle = np.arccos(scalar_proj) 

        # print rectangle's angle of rotation
        if self._crot:
            print(f'Rect\'s angle of rotation is: {self._rotAngle} degrees clockwise')
        else:
            print(f'Rect\'s angle of rotation is: {self._rotAngle} anticlockwise degrees')
        

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

    def gen_suitable(self):
        """
            Generates the proper set of coords
            which we can join with a line.
        """

        # Get unique y values
        all_yvals = self._contours[0][:,1].copy()  # Get all y values in contour
        unique_vals, count = np.unique(all_yvals, return_counts=True)
        min_by = np.intp(self._center[1] - 0.5 * self._height)  # min. bounding box coord (yaxis)
        max_by = np.intp(self._center[1] + 0.5 * self._height)  # max. bounding box coord (yaxis)

        # Get suitable y values that repeat
        suitable_yval = unique_vals[count > 1]

        # filter suitable_yval to select values between min_by and max_by
        filter = (np.where((suitable_yval >= min_by) & (suitable_yval <= max_by)))[0]
        suitable_yval = suitable_yval[filter]

        # Get filter to select x values that correspond to suitable_yval
        xfilter = np.isin(self._contours[0][:,1], suitable_yval)

        # Get corresponding x coords to suitable_yval
        self._suitable_coords = self._contours[0][xfilter]

    def var_width(self):
        prev = 0  # used to track the prev point's y coord drawn on img
        seen = []
        sep = 25 # determines the separation or spacing for arrowed lines

        # Get the varying widths
        for _, yval in self._suitable_coords:
            if yval - prev < sep:
                continue

            prev = yval

            if yval in seen:
                continue
            else:
                seen.append(yval)

            point = (np.where(self._contours[0][:,1] == yval))[0]
            if len(point) == 0:
                # skip rest of code if nothing is found
                continue

            xmin, xmax = np.inf, 0

            for xval_loc in point:
                xval = self._contours[0][xval_loc][0]
                if xval < xmin:
                    xmin = xval

                if xval > xmax:
                    xmax = xval

            # Calculate the width and store it
            width = xmax - xmin

            if width < 2:
                continue # skip small widths 

            self._varwidth.append(width)

            # Convert the lines to a proper basis and store it
            line = np.array([[xmin, xmax],[yval, yval]])
            self._suitable_line.append(line)

class Prosths(WidthVar):
    """
        This class draws the object contour, 
        bounding box and the varying width.
    """

    def __init__(self, image_path):
        super().__init__(image_path)

    def run(self):
        """
            This method sets up the entire
            detection of human body part,
            segmentation, measurements etc..
        """
        self.detect(self._img) # detect object
        self.segment() # segment object 
        if self._segment_flag:
            # show contours and and widths if contour was detected
            self.get_min_rectBox() # get min.area Rectangle 
            self.get_rot_img() # get rotated image if min. Rect is rotated 
            self.gen_suitable() # get suitable points/lines 
            self.var_width() # get all the varying widths 
            self.show()

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
            text_str = f'Width: {width}'
            text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_PLAIN, 1, 2)
            
            # Get x and y pos for text 
            x_pos = int(0.5*(x1+x2-text_size[0]))
            y_pos = int(y1+20)

            # Place text underneath arrowed line
            cv2.putText(self._img, text_str, (x_pos, y_pos),\
                         cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

    def show(self, segment=False):
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

        

a = Prosths('images/rot-cup.jpg')
a.run()

import cv2
import math
import numpy as np
from threading import Thread as TH
import time




def skin_detector(frame):

    min_YCrCb = np.array([0,140,85],np.uint8)
    max_YCrCb = np.array([240,180,130],np.uint8)
    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    skinMask = cv2.dilate(skinRegionYCrCb, kernel, iterations = 2)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)

    skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinMask)

    return skinYCrCb






class MakeArea:

    #global en cls
    
    def __init__(self, points, adding, landmarks, frame_head, mode_region, threshold):

        self.points      = points
        self.adding      = adding
        self.landmarks   = landmarks
        self.frame_head  = frame_head
        self.mode_region = mode_region
        self.threshold   = threshold


    @staticmethod
    def make_line(thresh):
        """We make line for detect more than one area
        with border, on eyelashes is paste to the border"""

        cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
        cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
        cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
        cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

        return thresh


    def recuperate_coordinates(self):
        """Here we recuperate coordinate of landmarks under convex area.
        middle mode's mean of 2 landmarks"""

        if self.mode_region == "middle":
            point  = self.points[-1]
            points = self.points[:-1]

        area_landmarks = [(self.landmarks.part(pts[0]).x + add[0],
                           self.landmarks.part(pts[1]).y + add[1])
                          for pts, add in zip(self.points, self.adding)]

        if self.mode_region == "middle":
            midle_point = self.landmarks.part(point[0]).x + self.landmarks.part(point[1]).x
            midle_point = int(midle_point / 2)
            area_landmarks += [(midle_point, self.landmarks.part(point[0]).y)]

        self.convexPoints = cv2.convexHull(np.array(area_landmarks))
        #cv2.drawContours(frame, [convexHull], -1, (0, 0, 255), 1)    

        crops = MakeArea.masks_from_convex(self)
        crop_threhsold, crop_frame, box_crop = crops

        return crop_threhsold, crop_frame, box_crop


    def masks_from_convex(self):
        """Make a mask of the region convex interest from the frame.
        Make a box of the mask"""

        height_frame, width_frame = self.frame_head.shape[:2]
        black_frame = np.zeros((height_frame, width_frame), np.uint8)
        mask = np.full((height_frame, width_frame), 255, np.uint8)
        cv2.fillPoly(mask, [self.convexPoints], (0, 0, 255))

        cv2.drawContours(self.frame_head, [self.convexPoints], -1, (0, 0, 255), 1)
        mask_threhsold = cv2.bitwise_not(black_frame, self.threshold.copy(), mask=mask)

        self.box_crop = cv2.boundingRect(self.convexPoints)
        x ,y, w, h = self.box_crop

        self.crop_threhsold = mask_threhsold[y:y+h, x:x+w]
        self.crop_threhsold = MakeArea.make_line(self.crop_threhsold)

        self.crop_frame     = self.frame_head[y:y+h, x:x+w]

        #cv2.rectangle(self.frame_head, (x, y), (x+w, y+h), (0, 0, 255), 1)

        return (self.crop_threhsold, self.crop_frame, self.box_crop)






class Detection:

    def __init__(self, data_points_wrinkle, data_feature_wrinkle,
                 landmarks, frame, mode_region, mode_feature, threshold,
                 crop_threshold, box_crop,
                 crop_frame, wrinkle_number):

        self.crop_threhsold  = crop_threhsold
        self.box_crop        = box_crop
        self.mode_region     = mode_region
        self.mode_feature    = mode_feature
        self.crop_frame      = crop_frame
        self.wrinkle_number  = wrinkle_number
        self.data_feature_wrinkle  = data_feature_wrinkle
        self.data_points_wrinkle   = data_points_wrinkle


    @staticmethod
    def extremums(c):
        """Recuperate left, right top and bottom extemums corner's"""

        xe = tuple(c[c[:, :, 0].argmin()][0])  #left
        ye = tuple(c[c[:, :, 1].argmin()][0])  #right
        we = tuple(c[c[:, :, 0].argmax()][0])
        he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

        return xe, ye, we, he




    def localisation_wrinkle(self):


        x ,y, w, h = self.box_crop

        wrinkles_list = []

        max_contour = int((w * h) * self.maxContour)
        min_contour = int((w * h) * self.minContour)

        maxLength = int(h * self.maxLength)
        minLength = int(h * self.minLength)

        contours, _ = cv2.findContours(self.crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in contours:

            if self.min_contour < cv2.contourArea(c) < self.max_contour:
                xe, ye, we, he = MakeArea.extremums(c)
                largeur  = we[0] - xe[0]
                longueur = he[1] - ye[1]

                if self.mode_feature == "length":
                     cls.condition_length(longueur, largeur, self.minLength,
                                         self.maxLength, self.crop_frame, he, ye, wrinkles_list)

                elif self.mode_feature == "width":
                    cls.condition_width(largeur, longueur, self.minWidth,
                                        self.maxWidth, self.crop_frame, xe, we, wrinkles_list)
       
                elif self.mode_feature == "lengthWidth":
                    cls.condition_lengthWidth(self.maxLength, longueur, self.minLength,
                                              self.maxWidth, largeur, self.minWidth,
                                              self.crop_frame, he, ye, wrinkles_list)

                elif self.mode_feature == "left":   
                    cv2.line(crop_frame, (we[0], ye[1]), (xe[0], he[1]), (0, 255, 0), 1)

                elif self.mode_feature == "right":
                    cv2.line(crop_frame, (xe[0], ye[1]), (we[0], he[1]), (0, 255, 255), 1)

                elif self.mode_feature == "length_or_width":
                    cls.condition_length_width(longueur, self.minLength, largeur,
                                               self.crop_frame, xe, ye, we, he, wrinkles_list)


        if len(wrinkles_list) >= self.wrinkle_number:
            [cv2.line(crop_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkles_list]



    @classmethod
    def condition_length(cls, longueur, largeur, minLength,
                         maxLength, crop_frame, he, ye, wrinkles):
        """Recuperate wrinkle if length > width"""

        if longueur > largeur and\
           minLength < longueur < maxLength:
            wrinkles.append((he, ye))

    @classmethod
    def condition_width(cls, largeur, longueur, minWidth,
                        maxWidth, crop_frame, xe, we):
        """Recuperate wrinkle if width > length"""

        if largeur   > longueur and\
            minWidth < largeur < maxWidth:
            cv2.line(crop_frame, xe, we, (0, 0, 255), 1)

    @classmethod
    def condition_lengthWidth(cls, maxLength, longueur, minLength, maxWidth,
                              largeur, minWidth, crop_frame, he, ye):

        """Recuperate wrinkle if width's and length are in interval."""

        if maxLength > longueur > minLength and\
            maxWidth > largeur  > minWidth:
            cv2.line(crop_frame, he, ye, (0, 0, 255), 1)

    @classmethod
    def condition_length_width(cls, longueur, minLength, largeur,
                               crop_frame, xe, ye, we, he, wrinkles):

        """Recuperate wrinkle if width > length
        and length > width and < to minLength."""

        if largeur > longueur:
            wrinkles.append((we, xe))

        elif longueur > largeur and longueur < minLength:
            wrinkles.append((he, ye))




#------- Raise anatomy --------

class Raise_region:
    """Here we raise region on threshold picture.
    Thank to that we can contourn false detections from
    anatomy part of the face.

    For that we must determin region from DLIB points for have
    coordinates. Sometimes we must increment region from the
    head region.

    Indeed we recuperate feature of the wrinkle from the treshold
    picture.
    
    """

    def __init__(self, landmarks, picture, head_box,
                 points_list, adding_height, add_width):

        """Importing landmarks from DLIB, the picture (threshold),
        who can be adaptativ (11, 2), (3, 5),
        the head box include in a box,
        our list points choosen from DLIB,
        our increment height and width regions"""

        self.landmarks = landmarks
        self.picture = picture
        self.head_box = head_box
        self.points_list = points_list
        self.adding_height = adding_height
        self.add_width = add_width

    def raising_part(self):
        """Put white on region convex interest on a gray picture"""

        #Add height px of our y points.
        width, height = self.head_box[2:]
        add_height_to_points = int(height * self.adding_height)
        add_width_to_points  = int(width  * self.add_width)
        
        #Recuperate landmarks 1:-1
        region = [(self.landmarks.part(n).x, self.landmarks.part(n).y - add_height_to_points)
                  for n in self.points_list[1: -1]]

        #First and last landmark (for hide on eyes)
        region1 = [(self.landmarks.part(self.points_list[0]).x + add_width_to_points,
                    self.landmarks.part(self.points_list[0]).y)]
        region2 = [(self.landmarks.part(self.points_list[-1]).x + add_width_to_points,
                    self.landmarks.part(self.points_list[-1]).y)]

        #Make one list
        region1 += region
        region1 += region2

        #Transfor points into array
        region = np.array(region1)
        #Fill the region in white color on a gray picture
        cv2.fillPoly(self.picture, [region], (255, 255, 255))
        
        








#Our landmarks points from DLIB.
ON_LEFT_EYE  = [17, 18, 19, 20, 21]
ON_RIGHT_EYE = [22, 23, 24, 25, 26]
MOUSE        = [21, 6, 10, 22]
LEFT_EYE     = [36, 37, 38, 39, 40, 41]
RIGHT_EYE    = [42, 43, 44, 45, 46, 47]
add = 0.055
zero = 0
def raising(landmarks, th, th1, head_box):

    global ON_LEFT_EYE
    global ON_RIGHT_EYE
    global MOUSE
    global LEFT_EYE
    global RIGHT_EYE

    global add
    global zero

  
    start_time_timmer = time.time()

##    raising_on_eyes1 = TH(target=Raise_region(landmarks, th, head_box,
##                                              ON_LEFT_EYE, add, zero).raising_part).start()
##
##    raising_on_eyes2 = TH(target=Raise_region(landmarks, th, head_box,
##                                              ON_RIGHT_EYE, add, zero).raising_part).start()
##
##    raising_mouse = TH(target=Raise_region(landmarks, th, head_box,
##                                              MOUSE, zero, zero).raising_part).start()
##
##    raising_eye1  = TH(target=Raise_region(landmarks, th1, head_box,
##                                              LEFT_EYE, zero, zero).raising_part).start()
##
##    raising_eye2  = TH(target=Raise_region(landmarks, th1, head_box,
##                                              RIGHT_EYE, add, zero).raising_part).start()
##
##    raising_eye3  = TH(target=Raise_region(landmarks, th, head_box,
##                                              LEFT_EYE, zero, zero).raising_part).start()
##
##    raising_eye4  = TH(target=Raise_region(landmarks, th, head_box,
##                                              RIGHT_EYE, zero, -0.5).raising_part).start()



    raising_on_eyes1 = Raise_region(landmarks, th, head_box, ON_LEFT_EYE, add, zero).raising_part()


    raising_on_eyes2 =Raise_region(landmarks, th, head_box,ON_RIGHT_EYE, add, zero).raising_part()


    raising_mouse = Raise_region(landmarks, th, head_box,MOUSE, zero, zero).raising_part()

    raising_eye1  = Raise_region(landmarks, th1, head_box,LEFT_EYE, zero, zero).raising_part()

    raising_eye2  = Raise_region(landmarks, th1, head_box,RIGHT_EYE, add, zero).raising_part()

    raising_eye3  = Raise_region(landmarks, th, head_box,LEFT_EYE, zero, zero).raising_part()

    raising_eye4  = Raise_region(landmarks, th, head_box,RIGHT_EYE, zero, -0.5).raising_part()



    print("raising: ", time.time() - start_time_timmer)





def detect_wringle(width_head, height_head, landmarks_head, frame_head, threshold):


    start_time_timmer = time.time()

    zero = 0
    mode_region_no         = ""
    mode_region_midddle    = "middle"

    mode_feature_length           = "length"
    mode_feature_width            = "width"
    mode_feature_lengthWidth      = "lengthWidth"
    mode_feature_left             = "left"
    mode_feature_right            = "right"
    mode_feature_length_or_width  = "length_or_width"


    addHeightBeetween = int(height_head * 0.09)  #5 de 74
    addWidthBeetween  = int(width_head  * 0.015) #1 de 90
    addingBeetween = ((-addWidthBeetween, -addHeightBeetween),
                      (addWidthBeetween, -addHeightBeetween),
                      (0, -addHeightBeetween))
    pointsBeetween = ((21, 21), (22, 22), (27, 27))


    addingCrow = ( (0, 0), (0, 0), (0, 0) )
    pointsCrowRight = ( (36, 37), (17, 37), (36, 31), (0, 17) )
    pointsCrowLeft  = ( (45, 44), (26, 44), (45, 35), (16, 26) )


    addWidthSM1 = int(width_head * 0.12) # 10 de 87 x1
    addWidthSM2 = int(width_head * 0.18) # 15 de 87 x2
    addingSideMouth1 = ( (addWidthSM1, 0), (-addWidthSM2, 0), (-addWidthSM2, 0) )
    addingSideMouth2 = ( (-addWidthSM1, 0), (addWidthSM2, 0), (addWidthSM2, 0) )
    pointsSideMouthRight = ( (30, 30), (14, 14), (12, 12) )
    pointsSideMouthReft  = ( (30, 30), (2, 2), (4, 4) )



    addHeightUnderEyes = int(height_head * 0.1) #8 de 85
    addingUnderEyes  = ((0, addHeightUnderEyes), (0, addHeightUnderEyes),
                       (0, 2 * addHeightUnderEyes), (0, 2 * addHeightUnderEyes))

    pointsUnderRight = ( (36, 36), (39, 39), (36, 36), (39, 39) )
    pointsUnderLeft  = ( (42, 42), (45, 45), (42, 42), (45, 45) )


    beetween_eyes = MakeArea(pointsBeetween, addingBeetween, landmarks_head,
                             frame_head, mode_region_no, threshold)
    beetween_eyes.recuperate_coordinates()


    crowRight = MakeArea(pointsCrowRight, addingCrow, landmarks_head, frame_head,
                         mode_region_midddle, threshold)
    crowRight.recuperate_coordinates()


    crowLeft = MakeArea(pointsCrowLeft, addingCrow, landmarks_head, frame_head,
                        mode_region_midddle, threshold)
    crowLeft.recuperate_coordinates()


    SideMouthRight = MakeArea(pointsSideMouthRight, addingSideMouth1, landmarks_head,
                              frame_head, mode_region_no, threshold)
    SideMouthRight.recuperate_coordinates()


    SideMouthLeft = MakeArea(pointsSideMouthReft, addingSideMouth2, landmarks_head,
                             frame_head, mode_region_no, threshold)
    SideMouthLeft.recuperate_coordinates()


    UnderEyeRight = MakeArea(pointsUnderRight, addingUnderEyes, landmarks_head,
                             frame_head, mode_region_no, threshold)
    UnderEyeRight.recuperate_coordinates()


    UnderEyeLeft = MakeArea(pointsUnderLeft, addingUnderEyes, landmarks_head,
                            frame_head, mode_region_no, threshold)
    UnderEyeLeft.recuperate_coordinates()


    print("area: ", time.time() - start_time_timmer)








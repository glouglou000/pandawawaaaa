import cv2
import math
import numpy as np
from threading import Thread as TH


def make_line(thresh):
    """We make line for detect more than one area
    with border, on eyelashes is paste to the border"""

    cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
    cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

    return thresh


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



def extremums(c):
    """Recuperate left, right top and bottom extemums corner's"""

    xe = tuple(c[c[:, :, 0].argmin()][0])  #left
    ye = tuple(c[c[:, :, 1].argmin()][0])  #right
    we = tuple(c[c[:, :, 0].argmax()][0])
    he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

    return xe, ye, we, he



def recuperate_coordinates(points, adding, landmarks, frame, mode):
    """Here we recuperate coordinate of landmarks under convex area.
    middle mode's mean of 2 landmarks"""

    if mode == "middle":
        point  = points[-1]
        points = points[:-1]

    area_landmarks = [(landmarks.part(pts[0]).x + add[0],
                       landmarks.part(pts[1]).y + add[1])

                      for pts, add in zip(points, adding)]

    if mode == "middle":
        midle_point = landmarks.part(point[0]).x + landmarks.part(point[1]).x
        midle_point = int(midle_point / 2)
        area_landmarks += [(midle_point, landmarks.part(point[0]).y)]


    convexHull = cv2.convexHull(np.array(area_landmarks))
    #cv2.drawContours(frame, [convexHull], -1, (0, 0, 255), 1)    

    return convexHull


def masks_from_convex(convexPoints, threshold, frame):
    """Make a mask of the region convex interest from the frame.
    Make a box of the mask"""

    height_frame, width_frame = frame.shape[:2]
    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [convexPoints], (0, 0, 255))

    cv2.drawContours(frame, [convexPoints], -1, (0, 0, 255), 1)
    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask)

    box_crop = cv2.boundingRect(convexPoints)
    x ,y, w, h = box_crop

    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame[y:y+h, x:x+w]

    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return crop_threhsold, crop_frame, box_crop



def condition_length(longueur, largeur, minLength,
                     maxLength, crop_frame, he, ye, wrinkles):
    """Recuperate wrinkle if length > width"""

    if longueur > largeur and\
       minLength < longueur < maxLength:
        wrinkles.append((he, ye))


def condition_width(largeur, longueur, minWidth,
                    maxWidth, crop_frame, xe, we, wrinkles):
    """Recuperate wrinkle if width > length"""

    if largeur > longueur and\
        minWidth < largeur < maxWidth:
        cv2.line(crop_frame, xe, we, (0, 0, 255), 1)


def condition_lengthWidth(maxLength, longueur, minLength, maxWidth,
                          largeur, minWidth, crop_frame, he, ye, wrinkles):

    """Recuperate wrinkle if width's and length are in interval."""

    if maxLength > longueur > minLength and\
        maxWidth > largeur > minWidth:
        cv2.line(crop_frame, he, ye, (0, 0, 255), 1)


def condition_length_width(longueur, minLength, largeur,
                           crop_frame, xe, ye, we, he, wrinkles_list):

    """Recuperate wrinkle if width > length
    and length > width and < to minLength."""

    if largeur > longueur:
        wrinkles_list.append((we, xe))

    elif longueur > largeur and longueur < minLength:
        wrinkles_list.append((he, ye))



def localisation_wrinkle(crop_threhsold, box_crop,
                         minContour, maxContour, minLength,
                         maxLength, minWidth, maxWidth, mode, crop_frame, wrinkle_number):


    x ,y, w, h = box_crop

    wrinkles_list = []

    max_contour = int((w * h) * maxContour)
    min_contour = int((w * h) * minContour)

    maxLength = int(h * maxLength)
    minLength = int(h * minLength)

    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:


        if min_contour < cv2.contourArea(c) < max_contour:
            xe, ye, we, he = extremums(c)
            largeur  = we[0] - xe[0]
            longueur = he[1] - ye[1]

            if mode == "length":
                 condition_length(longueur, largeur, minLength,
                                  maxLength, crop_frame, he, ye, wrinkles_list)

            elif mode == "width":
                condition_width(largeur, longueur, minWidth,
                                maxWidth, crop_frame, xe, we, wrinkles_list)
   
            elif mode == "lengthWidth":
                condition_lengthWidth(maxLength, longueur, minLength, maxWidth,
                                     largeur, minWidth, crop_frame, he, ye, wrinkles_list)

            elif mode == "left":   
                cv2.line(crop_frame, (we[0], ye[1]), (xe[0], he[1]), (0, 255, 0), 1)

            elif mode == "right":
                cv2.line(crop_frame, (xe[0], ye[1]), (we[0], he[1]), (0, 255, 255), 1)

            elif mode == "length_or_width":
                condition_length_width(longueur, minLength, largeur,
                                       crop_frame, xe, ye, we, he, wrinkles_list)



    if len(wrinkles_list) >= wrinkle_number:
        [cv2.line(crop_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkles_list]




def wrinkle_function(head_box_head, data_points, data_feature, landmarks,
                     frame_head, threshold, mode1, mode2, number):

    _, _, width_head, height_head = head_box_head

    adding, points = data_points

    convexPoints = recuperate_coordinates(points, adding, landmarks, frame_head, mode1)
    crop_threhsold, crop_frame, box_crop = masks_from_convex(convexPoints, threshold, frame_head)


    maxContour, minContour, minLength, maxLength, minWidth, maxWidth = data_feature

    localisation_wrinkle(crop_threhsold, box_crop,
                         minContour, maxContour, minLength,
                         maxLength, minWidth, maxWidth, mode2, crop_frame, number)





#------- Raise anatomy --------
def raising_part(landmarks_head, picture, head_box_head, points_list, adding_height, add_width):
    """Put white on region interest on a gray picture"""

    #Add height px of our y points.
    width, height = head_box_head[2:]
    add_height_to_points = int(height * adding_height)
    add_width_to_points = int(width * add_width)
    
    #Recuperate landmarks 1:-1
    region = [(landmarks_head.part(n).x, landmarks_head.part(n).y - add_height_to_points)
              for n in points_list[1: -1]]

    #First and last landmark (for hide on eyes)
    region1 = [(landmarks_head.part(points_list[0]).x + add_width_to_points,
                landmarks_head.part(points_list[0]).y)]
    region2 = [(landmarks_head.part(points_list[-1]).x + add_width_to_points,
                landmarks_head.part(points_list[-1]).y)]

    #Make one list
    region1 += region
    region1 += region2

    #Transfor points into array
    region = np.array(region1)
    #Fill the region in white color on a gray picture
    cv2.fillPoly(picture, [region], (255, 255, 255))





def raising(landmarks, th, th1, head_box):

    #Our landmarks points.
    on_left_eye  = [17, 18, 19, 20, 21]
    on_right_eye = [22, 23, 24, 25, 26]
    mouse     = [21, 6, 10, 22]
    left_eye  = [36, 37, 38, 39, 40, 41]
    right_eye = [42, 43, 44, 45, 46, 47]

    #Call raising function from thead functions
    raising_on_eyes1 = TH(target=raising_part(landmarks, th, head_box, on_left_eye, 0.055, 0))# 5 91
    raising_on_eyes2 = TH(target=raising_part(landmarks, th,head_box, on_right_eye, 0.055, 0))
    raising_mouse = TH(target=raising_part(landmarks, th, head_box, mouse, 0, 0))
    raising_eye1 = TH(target=raising_part(landmarks, th1, head_box, left_eye, 0, 0))
    raising_eye2 = TH(target=raising_part(landmarks, th1, head_box, right_eye, 0, -0.5))
    raising_eye3 = TH(target=raising_part(landmarks, th, head_box, left_eye, 0, 0))
    raising_eye4 = TH(target=raising_part(landmarks, th, head_box, right_eye, 0, -0.5))

    #Start thread
    raising_on_eyes1.start()
    raising_on_eyes2.start()
    raising_mouse.start()
    raising_eye1.start()
    raising_eye2.start()
    raising_eye3.start()
    raising_eye4.start()

    #join thread
    raising_on_eyes1.join()
    raising_on_eyes2.join()
    raising_mouse.join()
    raising_eye1.join()
    raising_eye2.join()
    raising_eye3.join()
    raising_eye4.join()


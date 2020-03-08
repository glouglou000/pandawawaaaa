import cv2
import math
import numpy as np
import threading

#Les boucles qui ralentissent
#Les zones mals définient


def make_line(thresh):
    """We make line for detect more than one area
    with border, on eyelashes is paste to the border"""

    cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
    cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

    return thresh


def recuperate_coordinates(points, adding, landmarks, frame, mode):


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

    height_frame, width_frame = frame.shape[:2]
    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [convexPoints], (0, 0, 255))

    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask)

    box_crop = cv2.boundingRect(convexPoints)
    x ,y, w, h = box_crop

    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame[y:y+h, x:x+w]

    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    return crop_threhsold, crop_frame, box_crop



def masks_from_box(convexPoints, threshold, frame):

    box_crop = cv2.boundingRect(np.array(convexPoints))
    x ,y, w, h = box_crop

    crop_threhsold = threshold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)
 
    crop_frame      = frame[y:y+h, x:x+w]

    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return crop_threhsold, crop_frame, box_crop



def extremums(c):

    xe = tuple(c[c[:, :, 0].argmin()][0])  #left
    ye = tuple(c[c[:, :, 1].argmin()][0])  #right
    we = tuple(c[c[:, :, 0].argmax()][0])
    he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

    return xe, ye, we, he



def condition_length(longueur, largeur, minLength,
                     maxLength, crop_frame, he, ye, wrinkles):

    if longueur > largeur and\
       minLength < longueur < maxLength:
        wrinkles.append((he, ye))


def condition_width(largeur, longueur, minWidth,
                    maxWidth, crop_frame, xe, we, wrinkles):

    if largeur > longueur and\
        minWidth < largeur < maxWidth:
        cv2.line(crop_frame, xe, we, (0, 0, 255), 1)


def condition_lengthWidth(maxLength, longueur, minLength, maxWidth,
                          largeur, minWidth, crop_frame, he, ye, wrinkles):

    if maxLength > longueur > minLength and\
        maxWidth > largeur > minWidth:
        cv2.line(crop_frame, he, ye, (0, 0, 255), 1)


def condition_length_width(longueur, minLength, largeur,
                           crop_frame, xe, ye, we, he, wrinkles_list):

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
































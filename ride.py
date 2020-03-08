import cv2
import numpy as np
from threading import Thread as TH
import math

from dlib import get_frontal_face_detector, shape_predictor
from video_treatment import search_video_size

from paths import dlib_model
from paths import dlib_model1

from paths import video_path

from dlib_points.points_of_face import load_model_dlib

from dlib_points.points_of_face import head_points

from scipy.spatial import distance as dist



from figure.wrinkle.ride_utils import raising_part
from figure.wrinkle.ride_utils import wrinkle_function
from figure.wrinkle.ride_utils import skin_detector
from figure.wrinkle.ride_utils import raising



from figure.wrinkle.ride_utils import recuperate_coordinates
from figure.wrinkle.ride_utils import masks_from_convex
from figure.wrinkle.ride_utils import masks_from_box
from figure.wrinkle.ride_utils import localisation_wrinkle



predictor = ""
detector = ""

predictor1 = ""
detector1 = ""



def a(dlib_model):
    global predictor
    global detector
    predictor, detector = load_model_dlib(dlib_model)


def b(dlib_model1):
    global predictor1
    global detector1
    predictor1, detector1 = load_model_dlib(dlib_model1)

t1 = TH(target=a(dlib_model))
t2 = TH(target=b(dlib_model1))

t1.start()
t2.start()

t1.join()
t2.join()





from video_capture_utils.video_capture_utils import resize_face, resize_eyes

video = video_path.format("e.mp4")
cap = cv2.VideoCapture(video)


#face_division = search_video_size(video, predictor, detector, dlib_model, 93)
face_division = 2.899999999999998
face_division = 1.650000000000001




def wrinkle_crow_feet(zero):

    #crow feet 
    data_points_crow_right = ( ( (0, 0), (0, 0), (0, 0) ), ( (36, 37), (17, 37), (36, 31), (0, 17) ) )
    data_points_crow_left  = ( ( (0, 0), (0, 0), (0, 0) ), ( (45, 44), (26, 44), (45, 35), (16, 26) ) )
    data_feature_crow      = (0.8, 0.003, 0.26 , zero, 0.28, zero)


    #right.
    t1 = TH(target=wrinkle_function(head_box_head, data_points_crow_right, data_feature_crow,
                                landmarks_head, frame_head, threshold, "middle", "length_or_width", 4))

    #left.
    t2 = TH(target=wrinkle_function(head_box_head, data_points_crow_left, data_feature_crow,
                     landmarks_head, frame_head, threshold, "middle", "length_or_width", 4))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    

def wrinkle_beetween_eye(zero, width_head, height_head):

    #Beetween eyes.
    adding_height = int(height_head * 0.09)   #5 de 74
    adding_width  = int(width_head  * 0.015)  #1 de 90

    data_points_beetween_eye = ( ( (-adding_width, -adding_height), (adding_width, -adding_height),
                                   (0, -adding_height) ), ( (21, 21), (22, 22), (27, 27) ))

    data_feature_beetween_eye = (0.5, 0.075, zero, 100000, zero, zero)
    t1 = TH(target=wrinkle_function(head_box_head, data_points_beetween_eye, data_feature_beetween_eye,
                     landmarks_head, frame_head, threshold, "", "length", 2))

    t1.start()
    t1.join()



def wrinkle_side_mouth(zero, width_head):

    #side mouth
    add1 = int(width_head * 0.12) # 10 de 87 x1
    add2 = int(width_head * 0.18) # 15 de 87 x2

    data_points_side_mouth_right = ( ( (add1, 0), (-add2, 0), (-add2, 0) ), ( (30, 30), (14, 14), (12, 12)) )
    data_points_side_mouth_left  = ( ( (-add1, 0), (add2, 0), (add2, 0)  ), ( (30, 30), (2, 2), (4, 4) ) )

    data_feature_side_mouth      = (0.8, 0.008, 0.26, 10000, int(width_head * 0.08), 10000)

    #right.
    t1 = TH(target=wrinkle_function(head_box_head, data_points_side_mouth_right, data_feature_side_mouth,
                     landmarks_head, frame_head, threshold, "", "lengthWidth", 0))
    #left.
    t2 = TH(target=wrinkle_function(head_box_head, data_points_side_mouth_left, data_feature_side_mouth,
                     landmarks_head, frame_head, threshold, "", "lengthWidth", 0))

    t1.start()
    t2.start()

    t1.join()
    t2.join()



def wrinkle_under_eye(zero, height_head):
    add_height = int(height_head * 0.1) #8 de 85

    #under
    data_points_under_eye_left  = (((0, add_height), (0, add_height),
                                   (0, 2 * add_height), (0, 2 * add_height)),
                                   ((36, 36), (39, 39), (36, 36), (39, 39)))

    data_points_under_eye_right = (((0, add_height), (0, add_height),
                                   (0, 2 * add_height), (0, 2 * add_height)),
                                   ((42, 42), (45, 45), (42, 42), (45, 45)))

    data_feature_under_eyes  = (0.8, 0.05, zero, zero, zero, zero)

    #left.
    t1 = TH(target=wrinkle_function(head_box_head, data_points_under_eye_left, data_feature_under_eyes,
                     landmarks_head, frame_head, threshold, "", "left", 0))
    #right.
    t2 = TH(target=wrinkle_function(head_box_head, data_points_under_eye_right, data_feature_under_eyes,
                     landmarks_head, frame_head, threshold, "", "right", 0))

    t1.start()
    t2.start()

    t1.join()
    t2.join()























#------- Filters -----------
threshold = ""
threshold1 = ""
def threshold_filter(gray_head):

    global threshold
    global threshold1
    #Make an adaptative threhsold
    mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold = cv2.adaptiveThreshold(gray_head, 255, mode, cv2.THRESH_BINARY,11, 2)
    threshold1 = cv2.adaptiveThreshold(gray_head, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,3,5)


frame_skin = ""
frame_blur = ""
def skin_blur(frame_head):

    global frame_skin
    global frame_blur
    #Recuperate only skin color (raise hair)
    frame_skin = skin_detector(frame_head)
    frame_blur = cv2.GaussianBlur(frame_head, (5 ,5), 0)



#------- Landmarks -----------

landmarks_head = ""
head_box_head  = ""
#Find landmarks from models.
def landmarks62(gray_head, predictor, detector):

    global landmarks_head
    global head_box_head
    landmarks68 = [0, 17, 16, 8]
    landmarks_head, head_box_head = head_points(gray_head, predictor, detector, landmarks68)



landmarks_head1 = ""
head_box_head1  = "" 
def landmarks81(gray_head, predictor1, detector1):

    global landmarks_head1
    global head_box_head1
    landmarks81 = [68, 72, 16, 10, 6, 77]
    landmarks_head1, head_box_head1 = head_points(gray_head, predictor1, detector1, landmarks81)




while True:

    _, frame = cap.read()

    frame_head, gray_head = resize_eyes(frame, face_division)

    #---------Landmarks------------
    thread_landmarks1 = TH(target=landmarks62(gray_head, predictor, detector))
    thread_landmarks2 = TH(target=landmarks81(gray_head, predictor1, detector1))

    thread_landmarks1.start()
    thread_landmarks2.start()

    thread_landmarks1.join()
    thread_landmarks2.join()



    if landmarks_head is not None:

        #----------Filters------------
        filter1 = TH(target=threshold_filter(gray_head))
        filter2 = TH(target=skin_blur(frame_head))

        filter1.start()
        filter2.start()

        filter1.join()
        filter2.join()



        #Raising part
        raising(landmarks_head, threshold, threshold1, head_box_head)


        
        zero = 0
        _, _, width_head, height_head = head_box_head

##        th_wrinkle_crow = TH(target=wrinkle_crow_feet(zero))
##        th_beetween     = TH(target=wrinkle_beetween_eye(zero, width_head, height_head))
##        th_side_mouth   = TH(target=wrinkle_side_mouth(zero, width_head))
        th_under_eyes   = TH(target=wrinkle_under_eye(zero, height_head))


        #th_wrinkle_crow.start()
        #th_beetween.start()
        #th_side_mouth.start()
        th_under_eyes.start()

        #th_wrinkle_crow.join()
        #th_beetween.join()
        #th_side_mouth.join()
        th_under_eyes.join()


























##        """En faire des thread et supprimer la haut"""
##        def forehead(landmarks, landmarks81, head_box_head,
##                    head_box_head1, frame_head, threshold, threshold1):
##
##            #Without and with forehead head dimensions.
##            _, _, width_head, height_head = head_box_head
##            _, _, _, height_head1 = head_box_head1
##

##            #forehead
##            add1 = - int( (5 * height_head1) / 100)
##            adding = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
##                      (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
##                      (0, 0), (0, 0), (0, add1), (0, add1),
##                      (0, add1), (0, add1), (0, add1), (0, add1),
##                      (0, add1), (0, add1), (0, add1), (0, add1)]
##
##            points = [(75, 75), (76, 76), (68, 68), (69, 69),
##                      (70, 70), (71, 71), (80, 80), (72, 72),
##                      (73, 73), (79, 79), (74, 74), (26, 26),
##                      (25, 25), (24, 24), (23, 23), (22, 22),
##                      (21, 21), (20, 20), (19, 19), (18, 18),
##                      (17, 17)]
##
##            convexPoints = recuperate_coordinates(points, adding, landmarks, frame_head, "")
##            crop_frame, crop_threhsold, box_crop = masks_from_convex(convexPoints, threshold, frame_head)

##        t10 = threading.Thread(target=forehead(landmarks_head, landmarks_head1, head_box_head,
##                                              head_box_head1, frame_head, threshold, threshold1))
##        t10.join()





    cv2.imshow("frame_head", frame_head)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()






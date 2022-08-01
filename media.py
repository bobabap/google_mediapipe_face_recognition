import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import imutils
import math
from image_conversion import conversion
from url_img_load import url_img
import face_recognition
from sklearn import svm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
                                static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5)    

 

def detection():
    url_list, img_list = url_img('faces.txt') # url과 이미지가 담긴 리스트
    
    '''step 1'''
    '''profile image하나씩'''
    for file, img in zip(url_list[1:], img_list[1:]):
        profile_image, profile_landmark = conversion(img)

    

if __name__ == '__main__':
    detection()
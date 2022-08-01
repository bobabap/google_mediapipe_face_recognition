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


def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    
    return similarity


def detection():
    url_list, img_list = url_img('faces.txt') # url과 이미지가 담긴 리스트
    
    '''step 1'''
    '''profile image하나씩'''
    now_image = conversion(url_list[0], img_list[0])
    for file, img in zip(url_list[1:], img_list[1:]):
        profile_image = conversion(file, img)
        
        result = face_recognition.compare_faces([now_image[0]], profile_image[0])
        print(result)
        
        print(np.linalg.norm(np.asarray(now_image) - np.asarray(profile_image), ord=2))

    

if __name__ == '__main__':
    detection()
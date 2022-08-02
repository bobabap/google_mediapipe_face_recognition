from deformation import deformate
import face_recognition
import cv2
import numpy as np

if __name__ == '__main__':
    url_list, images = deformate('profiles/jihyo.txt')
    now_face = images[0]
    now_face_url = url_list[0]
    
    profile_image = images[1:]
    profile_url = url_list[1:]
    # width, height, _ = now_face.shape

    now_face = face_recognition.face_encodings(now_face)[0]
    # now_face = np.asarray(now_face)
    for u, p in zip(profile_url, profile_image):
        p = face_recognition.face_encodings(p)[0]
        # p = np.asarray(p)
        if len(p) < 0:
            print('No')
            continue
        
        
        
        # print(u, face_recognition.compare_faces([now_face], p))
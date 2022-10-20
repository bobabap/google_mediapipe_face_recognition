import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def guide_image(p_height, p_width): # 변형할 이미지의 shape
    IMAGE_FILES = os.listdir('guide/')
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread('guide/'+file)
            
            image = cv2.resize(image, (p_height, p_width), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            guide_image = image.copy()
            height, width = guide_image.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                x_min = width
                x_max = 0
                y_min = height
                y_max = 0
                for i in range(0, 468): # [10, 152], range(0, 468)
                    pt = face_landmarks.landmark[i]
                    x = int(pt.x * width)
                    y = int(pt.y * height)
                    
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y
                
                guide_image = cv2.resize(guide_image, (300, 300))
                guide_image = guide_image[y_min:y_max, x_min:x_max]

                results = face_mesh.process(guide_image)
                if not results.multi_face_landmarks:
                    continue
            
                height, width = guide_image.shape[:2]
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_x = int(face_landmarks.landmark[7].x * width) # 왼쪽 눈
                    left_eye_y = int(face_landmarks.landmark[7].y * height)
                    right_eye_x = int(face_landmarks.landmark[249].x * width) # 오른쪽 눈
                    right_eye_y = int(face_landmarks.landmark[249].y * height)
                    mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
                    mid_chin_y = int(face_landmarks.landmark[152].y * height)
                    mid_lib_x = int(face_landmarks.landmark[1].x * width) # 인중
                    mid_lib_y = int(face_landmarks.landmark[1].y * height)
                    nose_tip_x = int(face_landmarks.landmark[4].x * width) # 코끝
                    nose_tip_y = int(face_landmarks.landmark[4].y * height)
                    nose_mid_x = int(face_landmarks.landmark[6].x * width) # 중간
                    nose_mid_y = int(face_landmarks.landmark[6].y * width)
                    left_temple_x = int(face_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
                    left_temple_y = int(face_landmarks.landmark[21].y * height)
                    right_temple_x = int(face_landmarks.landmark[251].x * width) # 오른쪽 관자놀이
                    right_temple_y = int(face_landmarks.landmark[251].y * height)
                    left_cheek_x = int(face_landmarks.landmark[58].x * width) # 왼쪽 볼끝
                    left_cheek_y = int(face_landmarks.landmark[58].y * height)
                    right_cheek_x = int(face_landmarks.landmark[367].x * width) # 오른쪽 볼끝
                    right_cheek_y = int(face_landmarks.landmark[367].y * height)
                    mid_head_x = int(face_landmarks.landmark[10].x * width) # 미간
                    mid_head_y = int(face_landmarks.landmark[10].y * height)
            
                
    # cv2.imshow('guide_Image', guide_image)
    # cv2.waitKey(0)
            
    return nose_mid_x, nose_mid_y, mid_lib_x, mid_lib_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_tip_x, nose_tip_y, mid_chin_x, mid_chin_y, left_cheek_x, left_cheek_y, right_cheek_x, right_cheek_y, mid_head_x, mid_head_y ,left_temple_x, left_temple_y, right_temple_x, right_temple_y

from turtle import shape
import cv2
import mediapipe as mp
import numpy as np
import math
import os
from url_img_load import url_img
from sklearn import svm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def guide_image(p_height, p_width): # 변형할 이미지의 shape
    IMAGE_FILES = os.listdir('guied/')
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread('guied/'+file)
            
            image = cv2.resize(image, (p_height, p_width), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            height, width = annotated_image.shape[:2]
            
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
                
                annotated_image = annotated_image[y_min-50:y_max+50, x_min-50:x_max+50]
                
                results = face_mesh.process(annotated_image)
                if not results.multi_face_landmarks:
                    continue
            
                height, width = annotated_image.shape[:2]
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
            
                
    # cv2.imshow('guide_Image', annotated_image)
    # cv2.waitKey(0)
            
    return nose_mid_x, nose_mid_y, mid_lib_x, mid_lib_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_tip_x, nose_tip_y, mid_chin_x, mid_chin_y, left_cheek_x, left_cheek_y, right_cheek_x, right_cheek_y, mid_head_x, mid_head_y ,left_temple_x, left_temple_y, right_temple_x, right_temple_y



def deformate(files):
    url_list, face_list = url_img(files)
    
    image_list = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx, image in enumerate(face_list):
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            
            # print(image.shape) # 원본 shape
            # if annotated_image.shape[0] > 1500 or annotated_image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소 
            #     annotated_image = cv2.resize(annotated_image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LANCZOS4)
            # print('-->', image.shape) # 축소후 shape
            
            height, width = annotated_image.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:

                x_min = width
                x_max = 0
                y_min = height
                y_max = 0
                
                for i in range(0, 469):
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
                    # cv2.circle(annotated_image, (x,y), 1, (250, 5, 0), -1)
                    # cv2.putText(annotated_image, f"{i}" , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # annotated_image = annotated_image[y_min:y_max, x_min:x_max]
            
            cv2.imshow('Image', annotated_image)
            cv2.waitKey(0)
                    
            mid_head_x = int(face_landmarks.landmark[10].x * width) # 미간
            mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
            nose_tip_x = int(face_landmarks.landmark[4].x * height)
            left_temple_x = int(face_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
            right_temple_x = int(face_landmarks.landmark[251].x * width) # 오른쪽 관자놀이


            # 얼굴이 왼쪽 방향인 경우 플립
            if nose_tip_x < left_temple_x:
                annotated_image = cv2.flip(annotated_image, 1)
            else:
                if right_temple_x < nose_tip_x:
                    pass
                else:
                    if abs(mid_head_x - left_temple_x) < abs(right_temple_x - mid_head_x): # 왼쪽방향 얼굴 x축 반전
                        annotated_image = cv2.flip(annotated_image, 1)


            results = face_mesh.process(annotated_image)
            if not results.multi_face_landmarks:
                continue
            
            height, width = annotated_image.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                
                mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
                mid_chin_y = int(face_landmarks.landmark[152].y * height)
                mid_head_x = int(face_landmarks.landmark[10].x * width) # 미간
                mid_head_y = int(face_landmarks.landmark[10].y * height)

                tan_theta = (mid_chin_x - mid_head_x)/(mid_chin_y - mid_head_y)
                theta = np.arctan(tan_theta)
                rotate_angle = theta *180/math.pi
                image_center = tuple(np.array(annotated_image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D((image_center), -rotate_angle, 1.0)
                annotated_image  = cv2.warpAffine(annotated_image, rot_mat, annotated_image.shape[1::-1], flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))

                cv2.imshow('Image', annotated_image)
                cv2.waitKey(0)
                
                results = face_mesh.process(annotated_image)
                
                if not results.multi_face_landmarks:
                    continue
                
                height, width = annotated_image.shape[:2]
                for face_landmarks in results.multi_face_landmarks:
                    x_min = width
                    x_max = 0
                    y_min = height
                    y_max = 0
                    
                    for i in range(0, 469):
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
                            
                nose_tip_x = int(face_landmarks.landmark[1].x * width) # 인중
                nose_tip_y = int(face_landmarks.landmark[1].y * height)
                left_eye_x = int(face_landmarks.landmark[7].x * width) # 왼쪽 눈
                left_eye_y = int(face_landmarks.landmark[7].y * height)
                right_eye_x = int(face_landmarks.landmark[249].x * width) # 오른쪽 눈
                right_eye_y = int(face_landmarks.landmark[249].y * height)
                    
                
                g_nose_mid_x, g_nose_mid_y, g_mid_lib_x, g_mid_lib_y, g_left_eye_x, g_left_eye_y, g_right_eye_x, g_right_eye_y, g_nose_tip_x, g_nose_tip_y, g_mid_chin_x, g_mid_chin_y, g_left_cheek_x, g_left_cheek_y, g_right_cheek_x, g_right_cheek_y, g_mid_head_x, g_mid_head_y, g_left_temple_x, g_left_temple_y, g_right_temple_x, g_right_temple_y = guide_image(height, width)
                
                # 공간 확보, 기하학 변형전 평행이동
                M = np.float32([[1,0,-(g_nose_tip_x-nose_tip_x)],[0,1,-(g_nose_tip_y-nose_tip_y)]])
                annotated_image = cv2.warpAffine(annotated_image, M, (width - (g_nose_tip_x - nose_tip_x), height - (g_nose_tip_y - nose_tip_y)), flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))

                # cv2.imshow('Image', annotated_image)
                # cv2.waitKey(0)
                
                # ---1 변환 전, 후 각 3개의 좌표 생성
                pts1 = np.float32([[left_eye_x, left_eye_y],[right_eye_x, right_eye_y],[nose_tip_x, nose_tip_y]]) # 원래 삼각 좌표
                # '''미리 정면 랜드마트 불러오고 그 랜드마크로 모든 랜드마크 이동'''
                pts2 = np.float32([[g_left_eye_x, left_eye_y], [g_right_eye_x, right_eye_y], [g_nose_tip_x, nose_tip_y]]) # 변형 될 위치 삼각 좌표
                
                # #---3 짝지은 3개의 좌표로 변환 행렬 계산
                mtrx = cv2.getAffineTransform(pts1, pts2)  #####################################
                
                # #---4 어핀 변환 적용  민트색  
                annotated_image = cv2.warpAffine(annotated_image, mtrx, (int(width*1.5), height), flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))
                
                image_list.append(annotated_image)
                
                # cv2.imshow('Image', annotated_image)
                # cv2.waitKey(0)
    return url_list, image_list
        
if __name__ == '__main__':
    deformate()
import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
                                static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5)

def crop(image):

    result = face_mesh.process(image)
    
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        for i in range(0, 468): # [10, 152], range(0, 468)
            pt = facial_landmarks.landmark[i]
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
    
        # image = image[y_min-50 : y_max+50, x_min-50 : x_max+50]

    croped_image = image.copy()
    Reset_result = face_mesh.process(croped_image) # 랜드마크 재설정
    
    return croped_image, Reset_result

def rotate_img(image, result):
    
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0

        for i in range(0, 468): # [10, 152], range(0, 468)        
            pt = facial_landmarks.landmark[i]
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
                
        mid_forehead_X = facial_landmarks.landmark[10].x # 중앙 미간 x
        mid_forehead_Y = facial_landmarks.landmark[10].y # 중앙 미간 y
        mid_chin_X = facial_landmarks.landmark[152].x # 중앙 턱 x
        mid_chin_Y = facial_landmarks.landmark[152].y # 중앙 턱 y
        
        '''얼굴 수평 이동'''
        tan_theta = (mid_chin_X - mid_forehead_X)/(mid_chin_Y - mid_forehead_Y)
        theta = np.arctan(tan_theta)
        rotate_angle = theta *180/math.pi
        rot_mat = cv2.getRotationMatrix2D((height//2, width//2), -rotate_angle, 1.0)
        image  = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    
        rotate_image = image.copy()
        Reset_result = face_mesh.process(rotate_image) # 랜드마크 재설정
    
    return rotate_image, Reset_result



def centerd_img(image, result):
    
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0

        for i in range(0, 468): # [10, 152], range(0, 468)        
            pt = facial_landmarks.landmark[i]
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
        
        '''얼굴 중앙 이동'''
        move_x = int(width/2) - (int(x_min) + int(((x_max - x_min)/2)))
        move_y = int(height/2) - (int(y_min) + int(((y_max - y_min)/2)))
            
        M = np.float32([[1, 0, move_x], [0, 1, move_y]])
        
        image = cv2.warpAffine(image, M, (0, 0), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

        centerd_image = image.copy()
        Reset_result = face_mesh.process(centerd_image) # 랜드마크 재설정
        
    return centerd_image, Reset_result


def geometric_deformation(image, result):
    
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        
        mid_nose_x = int(facial_landmarks.landmark[4].x * width)
        mid_nose_y = int(facial_landmarks.landmark[4].y * height)
        
        for i in range(0, 468): # [10, 152], range(0, 468)
            pt = facial_landmarks.landmark[i]
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
    
    # 코 끝이 중앙으로 이동하면서 모든 랜드마크 이동
    dx, dy = 100, 50
    
    geometrically_modified = image.copy()
    Reset_result = face_mesh.process(geometrically_modified) # 랜드마크 재설정

    return image, Reset_result


def conversion(image):
    # print(image.shape)
    if image.shape[0] > 1500 or image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소
        image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # print('-->', image.shape)
    
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    original_image = image.copy()
    # cv2.imshow('Image', original_image)
    # cv2.waitKey(0)
    
    # crop
    croped_image, Reset_result = crop(original_image)
    # cv2.imshow('Image', croped_image)
    # cv2.waitKey(0)
    
    # rotate
    rotate_image, Reset_result = rotate_img(croped_image, Reset_result)
    # cv2.imshow('Image', rotate_image)
    # cv2.waitKey(0)
    
    # centerd
    centerd_image, Reset_result = centerd_img(rotate_image, Reset_result)
    # cv2.imshow('Image', centerd_image)
    # cv2.waitKey(0)
    
    # geometric_deformation
    final_image, final_result = geometric_deformation(centerd_image, Reset_result)
    
    
    
    height, width = final_image.shape[:2]

    landmark = []
    for facial_landmarks in final_result.multi_face_landmarks:
        
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        
        for i in range(0, 468): # [10, 152], range(0, 468)
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * width)
            y = int(pt.y * height)
            landmark.append((pt.x, pt.y))
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y  
                                  
            # cv2.circle(final_image, (x,y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB, 점 색상
    
    
    # #얼굴이 있는 최 상하좌우 점
    # cv2.circle(image, (mid_nose_x, mid_nose_y), 4, (0, 255, 255), -1)
    # cv2.circle(image, (x_max,y_min), 2, (255, 255, 0), -1)
    # cv2.circle(image, (x_min,y_max), 2, (255, 255, 0), -1)
    # #얼굴 박스
    # cv2.circle(final_image, (width//2, height//2), 2, (255, 255, 0), -1)
    # cv2.rectangle(final_image, (x_max,y_min), (x_min,y_max), (0, 50, 0), 1)
    
    cv2.imshow('Image', final_image)
    cv2.waitKey(0)
    
    return final_image, landmark




def video_conversion(image):
    # print(image.shape)
    if image.shape[0] > 1500 or image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소
        image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # print('-->', image.shape)
    
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    original_image = image.copy()
    # cv2.imshow('Image', original_image)
    # cv2.waitKey(0)
    
    # crop
    croped_image, Reset_result = crop(original_image)
    # cv2.imshow('Image', croped_image)
    # cv2.waitKey(0)
    
    # rotate
    rotate_image, Reset_result = rotate_img(croped_image, Reset_result)
    # cv2.imshow('Image', rotate_image)
    # cv2.waitKey(0)
    
    # centerd
    centerd_image, Reset_result = centerd_img(rotate_image, Reset_result)
    # cv2.imshow('Image', centerd_image)
    # cv2.waitKey(0)
    
    # geometric_deformation
    final_image, final_result = geometric_deformation(centerd_image, Reset_result)
    
    return final_image
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

'''크롭을 먼저하면 얼굴 인식하는데 방해됌 필요시 주석해제하고 사용'''
# def crop(image):
#     result = face_mesh.process(image)
#     height, width = image.shape[:2]
#     for facial_landmarks in result.multi_face_landmarks:
#         x_min = width
#         x_max = 0
#         y_min = height
#         y_max = 0
#         for i in range(0, 468): # 랜드마크 (x, y) 0부터 468  
#             pt = facial_landmarks.landmark[i]
#             x = int(pt.x * width)
#             y = int(pt.y * height)
#             if x < x_min:
#                 x_min = x
#             if x > x_max:
#                 x_max = x
#             if y < y_min:
#                 y_min = y
#             if y > y_max:
#                 y_max = y  
#     croped_image = image.copy()
#     Reset_result = face_mesh.process(croped_image) # 랜드마크 위치 갱신 (필수)
#     return croped_image, Reset_result

def rotate_img(image):
    result = face_mesh.process(image)
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0

        for i in range(0, 468): # 랜드마크 (x, y) 0부터 468      
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
                
        mid_forehead_X = facial_landmarks.landmark[9].x # 중앙 미간 x
        mid_forehead_Y = facial_landmarks.landmark[9].y # 중앙 미간 y
        mid_chin_X = facial_landmarks.landmark[152].x # 중앙 턱 x
        mid_chin_Y = facial_landmarks.landmark[152].y # 중앙 턱 y
        
        '''얼굴 수평 이동'''
        tan_theta = (mid_chin_X - mid_forehead_X)/(mid_chin_Y - mid_forehead_Y)
        theta = np.arctan(tan_theta)
        rotate_angle = theta *180/math.pi
        rot_mat = cv2.getRotationMatrix2D((height//2, width//2), -rotate_angle, 1.0)
        image  = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    
        rotate_image = image.copy()
        Reset_result = face_mesh.process(rotate_image) # 랜드마크 위치 갱신 (필수)
    
    return rotate_image, Reset_result



def centerd_img(image, result):
    
    height, width = image.shape[:2]
    
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0

        for i in range(0, 468): # 랜드마크 (x, y) 0부터 468        
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
        Reset_result = face_mesh.process(centerd_image) # 랜드마크 위치 갱신 (필수)
        
    return centerd_image, Reset_result


def geometric_deformation(image, result):
    
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
                
    image = image[y_min:y_max, x_min:x_max] # 얼굴을 자른 상태에서 변형하기위해 단 얼굴 자르면 다음 얼굴인식이 잘안됌 그래서 최종단계에서 잘라야함
    
    
    '''자른 이미지 갱신을 위해 한번더 face_mesh.process'''
    height, width = image.shape[:2]
    Reset_result = face_mesh.process(image) # 랜드마크 위치 갱신 (필수)
    
    landmark = []
    for facial_landmarks in Reset_result.multi_face_landmarks:
        
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        
        '''랜드마크 위치 x, y'''
        mid_lib_x = int(facial_landmarks.landmark[1].x * width) # 인중
        mid_lib_y = int(facial_landmarks.landmark[1].x * height)
        
        mid_nose_x = int(facial_landmarks.landmark[4].x * width) # 코끝
        mid_nose_y = int(facial_landmarks.landmark[4].y * height)
        
        mid_head_x = int(facial_landmarks.landmark[10].x * width) # 미간
        mid_head_y = int(facial_landmarks.landmark[10].y * height)

        mid_chin_x = int(facial_landmarks.landmark[152].x * width) # 턱끝
        mid_chin_y = int(facial_landmarks.landmark[152].y * height)
        
        left_eye_y = int(facial_landmarks.landmark[7].y * height) # 왼쪽 눈
        left_eye_x = int(facial_landmarks.landmark[7].x * width)
        
        right_eye_x = int(facial_landmarks.landmark[249].x * width) # 오른쪽 눈
        right_eye_y = int(facial_landmarks.landmark[249].y * height)
        
        left_cheek_x = int(facial_landmarks.landmark[58].x * width) # 왼쪽 볼끝
        left_cheek_y = int(facial_landmarks.landmark[58].y * height)
        
        right_cheek_x = int(facial_landmarks.landmark[367].x * width) # 오른쪽 볼끝
        right_cheek_y = int(facial_landmarks.landmark[367].y * height)
        
        left_temple_x = int(facial_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
        left_temple_y = int(facial_landmarks.landmark[21].y * height)
        
        right_temple_x = int(facial_landmarks.landmark[21].x * width) # 오른쪽 관자놀이
        right_temple_y = int(facial_landmarks.landmark[21].y * height)


        '''입술중간:1, 코끝:4, 왼쪽눈:7, 오른쪽눈:249, 미간:9, 이마끝:10, 왼쪽관자놀이:21, 오른쪽관자놀이:251 왼쪽 볼끝: 58, 오른쪽 볼끝:367 턱 중앙:152, '''
        for i in range(0, 468): # 랜드마크 (x, y) 0부터 468  
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
                
            '''주석 해제시 점 하나씩 확인'''
            # cv2.circle(image, (x,y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB, 점 색상
            # cv2.putText(image, str(i) , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # cv2.imshow('affin', image)
            # cv2.waitKey(0)

    
    '''이미지 기하학 변형 부분'''
    # cv2.circle(image, (mid_chin_x,mid_head_y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB, 점 색상
    # cv2.circle(image, (mid_head_x,mid_head_y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB, 점 색상
    # cv2.imshow('origin',image)
    # rows, cols = image.shape[:2]
    # 중앙으로 이동

    # ---① 변환 전, 후 각 3개의 좌표 생성
    # pts1 = np.float32([[mid_head_x, mid_head_y], [mid_nose_x, mid_nose_y], [mid_chin_x, mid_chin_y]])
    # pts2 = np.float32([[width//2, height//2 - (height//2-mid_head_y)], [width//2, height//2], [width//2, height//2 + (height//2-mid_chin_y)]])

    # # ---② 변환 전 좌표를 이미지에 표시
    # cv2.circle(image, (mid_head_x, mid_head_y), 5, (255,0), -1)
    # cv2.circle(image, (mid_nose_x, mid_nose_y), 5, (0,255,0), -1)
    # cv2.circle(image, (mid_chin_x, mid_chin_y), 5, (0,0,255), -1)

    # #---③ 짝지은 3개의 좌표로 변환 행렬 계산
    # mtrx = cv2.getAffineTransform(pts1, pts2)
    # #---④ 어핀 변환 적용
    # dst = cv2.warpAffine(image, mtrx, (int(cols*1.5), rows))

    # cv2.imshow('affin', dst) # 변형 확인
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # ----------------------------------------------------
    
    
    geometrically_modified = image.copy()
    '''최종 얼굴에 랜드마크 표시하기'''
    # # 얼굴이 있는 최 상하좌우 점
    # cv2.circle(geometrically_modified, (mid_nose_x, mid_nose_y), 4, (0, 255, 255), -1)
    # cv2.circle(geometrically_modified, (x_max,y_min), 2, (255, 255, 0), -1)
    # cv2.circle(geometrically_modified, (x_min,y_max), 2, (255, 255, 0), -1)
    # # # 얼굴 박스
    # cv2.circle(geometrically_modified, (width//2, height//2), 2, (255, 255, 0), -1)
    # cv2.rectangle(geometrically_modified, (x_max,y_min), (x_min,y_max), (0, 50, 0), 1)
    # cv2.imshow('Image', geometrically_modified)
    # cv2.waitKey(0)
    # ----------------------------------------------------
    
    return image, landmark


'''media.py에서 원본 profile image 하나씩 넣고 함수 순서대로 실행하는 부분'''
def conversion(image):
    # print(image.shape) # 원본 shape
    '''이미지가 너무 크면 오류남'''
    if image.shape[0] > 1500 or image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소 
        image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # print('-->', image.shape) # 축소후 shape
    
    '''실행 속도 높이기위해 BGR로 색 변형'''
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    original_image = image.copy()
    '''imshow 주석 해제시 단계별로 이미지 변형 확인 가능'''
    
    # rotate
    rotate_image, Reset_result = rotate_img(original_image)
    cv2.imshow('Image', rotate_image)
    cv2.waitKey(0)
    # ---------------------
    
    # centerd
    centerd_image, Reset_result = centerd_img(rotate_image, Reset_result)
    cv2.imshow('Image', centerd_image)
    cv2.waitKey(0)
    # ---------------------
    
    # geometric_deformation
    final_image, landmark = geometric_deformation(centerd_image, Reset_result)
    cv2.imshow('Image', final_image)
    cv2.waitKey(0)
    # ---------------------
    
    return final_image, landmark




def video_conversion(image):
    # print(image.shape)
    if image.shape[0] > 1500 or image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소
        image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # print('-->', image.shape)
    
    '''실행 속도 높이기위해 BGR로 색 변형'''
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    original_image = image.copy()

    # rotate
    rotate_image, Reset_result = rotate_img(original_image)
    
    # centerd
    centerd_image, Reset_result = centerd_img(rotate_image, Reset_result)
    
    # geometric_deformation
    final_image, _ = geometric_deformation(centerd_image, Reset_result)
    
    # 비디오에는 변형된 이미지만 전달
    return final_image
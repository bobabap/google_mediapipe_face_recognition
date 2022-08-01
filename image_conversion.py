import cv2
from cv2 import INTER_LINEAR
import mediapipe as mp
import numpy as np
import math
from url_img_load import url_img

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
        
        mid_nose_x = int(facial_landmarks.landmark[4].x * width) # 코끝
        left_temple_x = int(facial_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
        right_temple_x = int(facial_landmarks.landmark[251].x * width) # 오른쪽 관자놀이

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
        
        '''얼굴 중앙 이동 팁: x 좌표는 왼쪽에서 오른쪽으로 밀고 y 좌표는 위에서 아래로 민다'''
        move_x = int(width/2) - (int(x_min) + int(((x_max - x_min)/2)))
        move_y = int(height/2) - (int(y_min) + int(((y_max - y_min)/2)))
            
        M = np.float32([[1, 0, move_x], [0, 1, move_y]])
        
        image = cv2.warpAffine(image, M, (0, 0), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
        
        # 얼굴이 왼쪽 방향인 경우 플립
        if mid_nose_x - left_temple_x < right_temple_x - mid_nose_x: # 왼쪽방향 얼굴 x축 반전
            image = cv2.flip(image, 1)
            
        centerd_image = image.copy()
        Reset_result = face_mesh.process(centerd_image) # 랜드마크 위치 갱신 (필수)
        
    return centerd_image, Reset_result



def guide_image_landmark(p_height, p_width): # 변형할 이미지의 shape
    g_image = cv2.imread('guide_face.png', cv2.IMREAD_COLOR)

    g_height, g_width = g_image.shape[:2]
    
    g_image_result = face_mesh.process(g_image) # 랜드마크 위치 갱신 (필수)
    for g_facial_landmarks in g_image_result.multi_face_landmarks:
        g_x_min = g_width
        g_x_max = 0
        g_y_min = g_height
        g_y_max = 0
        
        for i in range(0, 468): # [10, 152], range(0, 468)
            g_pt = g_facial_landmarks.landmark[i]
            g_x = int(g_pt.x * g_width)
            g_y = int(g_pt.y * g_height)
            
            if g_x < g_x_min:
                g_x_min = g_x
            if g_x > g_x_max:
                g_x_max = g_x
            if g_y < g_y_min:
                g_y_min = g_y
            if g_y > g_y_max:
                g_y_max = g_y   
                
            cv2.circle(g_image, (g_x, g_y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB
    g_image = g_image[g_y_min-50:g_y_max+50, g_x_min-50:g_x_max+50]
    
    # -----------------------------------------------------------------
    
    g_image = cv2.resize(g_image, (p_width, p_height), interpolation=INTER_LINEAR)
    '''최종 가이드 사진 랜드마크 갱신'''
    Reset_result = face_mesh.process(g_image) # 랜드마크 위치 갱신 (필수)
    height, width = g_image.shape[:2]
    
    for facial_landmarks in Reset_result.multi_face_landmarks:
    
        '''랜드마크 위치 x, y'''
        '''입술중간:1, 코끝:4, 왼쪽눈:7, 오른쪽눈:249, 미간:9, 이마끝:10, 왼쪽관자놀이:21, 오른쪽관자놀이:251 왼쪽 볼끝: 58, 오른쪽 볼끝:367 턱 중앙:152, '''
        mid_nose_x = int(facial_landmarks.landmark[4].x * width) # 코끝
        mid_nose_y = int(facial_landmarks.landmark[4].y * height)
        
        left_eye_x = int(facial_landmarks.landmark[7].x * width) # 왼쪽 눈
        left_eye_y = int(facial_landmarks.landmark[7].y * height)
        
        right_eye_x = int(facial_landmarks.landmark[249].x * width) # 오른쪽 눈
        right_eye_y = int(facial_landmarks.landmark[249].y * height)
        
        mid_chin_x = int(facial_landmarks.landmark[152].x * width) # 턱끝
        mid_chin_y = int(facial_landmarks.landmark[152].y * height)
        
        # mid_lib_x = int(facial_landmarks.landmark[1].x * width) # 인중
        # mid_lib_y = int(facial_landmarks.landmark[1].x * height)
        
        # left_cheek_x = int(facial_landmarks.landmark[58].x * width) # 왼쪽 볼끝
        # left_cheek_y = int(facial_landmarks.landmark[58].y * height)
        
        # right_cheek_x = int(facial_landmarks.landmark[367].x * width) # 오른쪽 볼끝
        # right_cheek_y = int(facial_landmarks.landmark[367].y * height)
        
        # left_temple_x = int(facial_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
        # left_temple_y = int(facial_landmarks.landmark[21].y * height)
        
        # right_temple_x = int(facial_landmarks.landmark[251].x * width) # 오른쪽 관자놀이
        # right_temple_y = int(facial_landmarks.landmark[251].y * height)

        # mid_head_x = int(facial_landmarks.landmark[10].x * width) # 미간
        # mid_head_y = int(facial_landmarks.landmark[10].y * height)
        
        
    # print([[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [mid_nose_x, mid_nose_y]])
    # print('guide image shape:',g_image.shape)
    # cv2.imshow('g_image', g_image) # 가이드 이미지 확인
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # landmark = {'left_eye':[left_eye_x, left_eye_y], 'right_eye':[right_eye_x, right_eye_y], 'mid_nose':[mid_nose_x, mid_nose_y], 'mid_chin':[mid_chin_x, mid_chin_y]}
    
    return left_eye_x, left_eye_y, right_eye_x, right_eye_y, mid_nose_x, mid_nose_y, mid_chin_x, mid_chin_y



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
                
    image = image[y_min-50:y_max+50, x_min-50:x_max+50] # 얼굴을 자른 상태에서 변형하기위해 단 얼굴 자르면 다음 얼굴인식이 잘안됌 그래서 최종단계에서 잘라야함
    '''자른 이미지 갱신을 위해 한번더 face_mesh.process'''
    Reset_result = face_mesh.process(image) # 랜드마크 위치 갱신 (필수)
    height, width = image.shape[:2]
    # --------------------------------여기까지 프로필 이미지 랜드마크 준비 끝
    
    for facial_landmarks in Reset_result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0

        mid_nose_x = int(facial_landmarks.landmark[4].x * width) # 코끝
        mid_nose_y = int(facial_landmarks.landmark[4].y * height)

        mid_chin_x = int(facial_landmarks.landmark[152].x * width) # 턱끝
        mid_chin_y = int(facial_landmarks.landmark[152].y * height)
        
        left_eye_x = int(facial_landmarks.landmark[7].x * width) # 왼쪽 눈
        left_eye_y = int(facial_landmarks.landmark[7].y * height)
        
        right_eye_x = int(facial_landmarks.landmark[249].x * width) # 오른쪽 눈
        right_eye_y = int(facial_landmarks.landmark[249].y * height)


    # print('profile image shape:',image.shape)
    
    '''이미지 기하학 변형 부분'''
    # 가이드 얼굴 랜드마크 정면
    g_left_eye_x, g_left_eye_y, g_right_eye_x, g_right_eye_y, g_mid_nose_x, g_mid_nose_y, g_mid_chin_x, g_mid_chin_y = guide_image_landmark(height, width)
    
    cv2.imshow('image', image) # 변형 확인
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    M = np.float32([[1,0,-(g_mid_nose_x-mid_nose_x)],[0,1,-(g_mid_nose_y-mid_nose_y)]])
    image = cv2.warpAffine(image, M, (width - (g_mid_nose_x-mid_nose_x), height - (g_mid_nose_y-mid_nose_y)))
        
    cv2.imshow('image', image) # 변형 확인
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 중앙으로 이동
    # ---1 변환 전, 후 각 3개의 좌표 생성
    pts1 = np.float32([[left_eye_x,left_eye_y],[right_eye_x, right_eye_y],[mid_nose_x, mid_nose_y]]) # 원래 삼각 좌표
    '''미리 정면 랜드마트 불러오고 그 랜드마크로 모든 랜드마크 이동'''
    pts2 = np.float32([[g_left_eye_x, g_left_eye_y], [g_right_eye_x, g_right_eye_y], [g_mid_nose_x, g_mid_nose_y]]) # 변형 될 위치 삼각 좌표
    
    # # ---2 변환 전 좌표를 이미지에 표시 빨간색
    cv2.circle(image, (g_left_eye_x,left_eye_y), 3, (255, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (g_right_eye_x,right_eye_y), 3, (255, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (g_mid_chin_x,g_mid_chin_y), 3, (255, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (width//2, height//2), 3, (255, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.rectangle(image, (x_max,y_min), (x_min,y_max), (255, 0, 255), 1)
    
    #---3 짝지은 3개의 좌표로 변환 행렬 계산
    mtrx = cv2.getAffineTransform(pts1, pts2)  #####################################
    
    #---4 어핀 변환 적용  민트색  
    image = cv2.warpAffine(image, mtrx, (int(width - (g_mid_nose_x-mid_nose_x)), height - (g_mid_nose_y-mid_nose_y))) #########################################
    # 변환 후 좌표를 이미지에 표시 민트색
    cv2.circle(image, (left_eye_x,left_eye_y), 4, (0, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (right_eye_x,right_eye_y), 4, (0, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (mid_nose_x,mid_nose_y), 4, (0, 0, 255), -1) # 원을 그릴 위치(x,y), 점 크기, (100, 100, 0)RGB
    cv2.circle(image, (width//2, height//2), 6, (255, 255, 0), -1)
    cv2.rectangle(image, (x_max,y_min), (x_min,y_max), (255, 255, 0), 1)
    
    image = cv2.resize(image, (width, height), interpolation=INTER_LINEAR) ########################################
    cv2.imshow('affin', image) # 변형 확인
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #----------------------------------------------------
    for facial_landmarks in result.multi_face_landmarks:
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        for i in range(0, 468): # [10, 152], range(0, 468) ##############################################
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * width)
            y = int(pt.y * height)
            
            cv2.circle(image, (x, y), 1, (250, 5, 0), -1)
            
    image = cv2.resize(image, (300, 300), interpolation=INTER_LINEAR) #################################################
    cv2.imshow('affin_test', image) # 변형 확인
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #----------------------------------------------------
    
    geometrically_modified = image.copy()
    '''최종 얼굴에 랜드마크 표시하기'''
    # # 얼굴이 있는 최 상하좌우 점
    cv2.circle(geometrically_modified, (mid_nose_x, mid_nose_y), 4, (0, 255, 255), -1)
    cv2.circle(geometrically_modified, (x_max,y_min), 2, (255, 255, 0), -1)
    cv2.circle(geometrically_modified, (x_min,y_max), 2, (255, 255, 0), -1)
    # # 얼굴 박스
    cv2.circle(geometrically_modified, (width//2, height//2), 2, (255, 255, 0), -1)
    cv2.rectangle(geometrically_modified, (x_max,y_min), (x_min,y_max), (0, 50, 0), 1)
    cv2.imshow('Image', geometrically_modified)
    cv2.waitKey(0)
    # ----------------------------------------------------
    # final_image = dst.copy()
    final_landmark = []
    return image, final_landmark


'''media.py에서 원본 profile image 하나씩 넣고 함수 순서대로 실행하는 부분'''
def conversion(image):
    # print(image.shape) # 원본 shape
    '''이미지가 너무 크면 오류남'''
    if image.shape[0] > 1500 or image.shape[1] > 1500: # 너비 높이 1500 이상이면 보간법으로 이미지 축소 
        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    # print('-->', image.shape) # 축소후 shape
    
    '''실행 속도 높이기위해 BGR로 색 변형'''
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    original_image = image.copy()
    '''imshow 주석 해제시 단계별로 이미지 변형 확인 가능'''
    
    # rotate
    rotate_image, Reset_result = rotate_img(original_image)
    # cv2.imshow('Image', rotate_image)
    # cv2.waitKey(0)
    # ---------------------
    
    # centerd
    centerd_image, Reset_result = centerd_img(rotate_image, Reset_result)
    # cv2.imshow('Image', centerd_image)
    # cv2.waitKey(0)
    # ---------------------
    
    # geometric_deformation
    final_image, landmark = geometric_deformation(centerd_image, Reset_result)
    # cv2.imshow('Image', final_image)
    # cv2.waitKey(0)
    # ---------------------
    
    return final_image, landmark
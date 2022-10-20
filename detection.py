import cv2
import mediapipe as mp
import os
import sys
import numpy as np
import math
from guide import guide_image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import time
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
import face_recognition    
from skimage.metrics import structural_similarity as ssim
from img2vec_pytorch import Img2Vec

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
    

def D(file):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        '''------------------------set process-------------------------'''
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            print("worning")
        annotated_image = image.copy() 

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
        
        '''------------------------imshow---------------------------'''
        # cv2.imshow('Image', annotated_image)
        # cv2.waitKey(0)
        annotated_image = annotated_image[y_min-10:y_max+10, x_min-10:x_max+10]

        '''---------------Symmetric alignment---------------'''
        mid_head_x = int(face_landmarks.landmark[10].x * width) # 미간
        mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
        nose_tip_x = int(face_landmarks.landmark[4].x * height) # 코끝
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
        '''------------------------reset process-------------------------'''
        results = face_mesh.process(annotated_image)
        if not results.multi_face_landmarks:
            print("worning")
                    
        height, width = annotated_image.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            
            mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
            mid_chin_y = int(face_landmarks.landmark[152].y * height)
            mid_head_x = int(face_landmarks.landmark[10].x * width) # 미간
            mid_head_y = int(face_landmarks.landmark[10].y * height)

            '''---------------Image Rotation---------------'''
            tan_theta = (mid_chin_x - mid_head_x)/(mid_chin_y - mid_head_y)
            theta = np.arctan(tan_theta)
            rotate_angle = theta *180/math.pi
            image_center = tuple(np.array(annotated_image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D((image_center), -rotate_angle, 1.0)
            annotated_image  = cv2.warpAffine(annotated_image, rot_mat, annotated_image.shape[1::-1], flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))
            '''------------------------imshow---------------------------'''
            # cv2.imshow('Image', annotated_image)
            # cv2.waitKey(0)
            annotated_image = cv2.resize(annotated_image, (300, 300))

            '''------------------------reset process-------------------------'''
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
            


            
            left_temple_x = int(face_landmarks.landmark[21].x * width) # 왼쪽 관자놀이
            left_temple_y = int(face_landmarks.landmark[21].y * height)
            right_temple_x = int(face_landmarks.landmark[251].x * width) # 오른쪽 관자놀이
            right_temple_y = int(face_landmarks.landmark[251].y * height)    
            left_cheek_x = int(face_landmarks.landmark[58].x * width) # 왼쪽 볼끝
            left_cheek_y = int(face_landmarks.landmark[58].y * height)
            right_cheek_x = int(face_landmarks.landmark[367].x * width) # 오른쪽 볼끝
            right_cheek_y = int(face_landmarks.landmark[367].y * height)
            
            g_nose_mid_x, g_nose_mid_y, g_mid_lib_x, g_mid_lib_y, g_left_eye_x, g_left_eye_y, g_right_eye_x, g_right_eye_y, g_nose_tip_x, g_nose_tip_y, g_mid_chin_x, g_mid_chin_y, g_left_cheek_x, g_left_cheek_y, g_right_cheek_x, g_right_cheek_y, g_mid_head_x, g_mid_head_y, g_left_temple_x, g_left_temple_y, g_right_temple_x, g_right_temple_y = guide_image(height, width)


            # [x,y] 좌표점을 4x2의 행렬로 작성
            # 좌표점은 좌상->좌하->우상->우하
            pts1 = np.float32([[left_temple_x, left_temple_y],[left_cheek_x, left_cheek_y],[right_temple_x, right_temple_y],[right_cheek_x, right_cheek_y]])

            # 좌표의 이동점
            pts2 = np.float32([[g_left_temple_x, g_left_temple_y],[g_left_cheek_x, g_left_cheek_y],[g_right_temple_x, g_right_temple_y],[g_right_cheek_x, g_right_cheek_y]])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            annotated_image = cv2.warpPerspective(annotated_image, M, (200,180), flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))
            
            
            # cv2.imshow('Image', annotated_image)
            # cv2.waitKey(0)



            '''------------------------imshow---------------------------'''
            # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
                        
            cv2.imshow('Image', annotated_image)
            cv2.waitKey(0)
            
    return annotated_image



# def run():
#     landmark_list = [D(img) for img in glob.glob('./image data/minjung/*.jpg')]
    
    
#     for l in landmark_list:
#         print(l)
#         print('-------------------------------------------------------------')


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

from scipy.stats import pearsonr
def pearson_similarity(a, b):
    return np.dot((a - np.mean(a)), (b - np.mean(b))) / ((np.linalg.norm(a - np.mean(a))) * (np.linalg.norm(b - np.mean(b))))
'''-------run-------'''


img2vec = Img2Vec(cuda=(torch.cuda.is_available()))
def img2vec(img):
    # Initialize Img2Vec with GPU
    img2vec = Img2Vec(cuda=True)

    # Get a vector from img2vec, returned as a torch FloatTensor
    vec = img2vec.get_vec(img, tensor=True)


def run():
    
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # img_list = [D(img) for img in glob.glob('./image data/minjung/*.jpg')]
    
    # IMAGE_FILES = os.listdir('./image data/minjung/')

    # A = img_list[0]
    # A_name = IMAGE_FILES[0]
    # for B, name in zip(img_list[1:], IMAGE_FILES[1:]):
    #     score = cosine_similarity(img2vec(A), img2vec(B)) # .flatten()
    #     print(f"{A_name}와 {name}의 Similarity: {score}")
'''------------------------------------------'''

   

if __name__ == '__main__':
    run()
    
    





    # 저장시
    # for n, i in enumerate(imgs):
    #     cv2.imwrite(f'crop_{n}.jpg', i)
    
''' ORB 기술자를 사용한 특징점 비교'''
    # for image in img_list[1:]:
    #     img1 = img_list[0] # queryImage
    #     img2 = image # trainImage

    #     sift = cv2.xfeatures2d.SIFT_create()
    #     kp1, des1 = sift.detectAndCompute(img1,None)
    #     kp2, des2 = sift.detectAndCompute(img2,None)
    #     bf = cv2.BFMatcher()

    #     matches = bf.knnMatch(des1,des2, k=2)
    #     good = []

    #     for m,n in matches:
    #         # if m.distance < 0.3*n.distance:
    #         if m.distance < n.distance:

    #             good.append([m])
    #     print('same_good', good)
    #     img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    #     plt.imshow(img3),plt.show()
    #     cv2.imshow("comapre", img3)
    #     cv2.waitKey(1)
    
    
'''랜드마크 저장'''
            # results = face_mesh.process(annotated_image)
            
            # if not results.multi_face_landmarks:
            #     continue
            
            # landmark = []
            
            # height, width = annotated_image.shape[:2]
            # for face_landmarks in results.multi_face_landmarks:
            #     x_min = width
            #     x_max = 0
            #     y_min = height
            #     y_max = 0
                
            #     for i in range(0, 469):
            #         pt = face_landmarks.landmark[i]
            #         x = pt.x
            #         y = pt.y
            #         z = pt.z
                    
            #         landmark.append([x,y,z])


    
'''dlib의 face_recognition 을 사용한 점수값'''
'''
        이거보다 더 좋은 결과를 얻을 수 있게 설계

    imgs = [cv2.imread(img, cv2.IMREAD_COLOR) for img in glob.glob('./image data/minjung/*.jpg')]
    img_names = os.listdir('./image data/minjung/')
    
    encodings = []
    for img in imgs:
        encodings.append(face_recognition.face_encodings(img)[0])
    
    print(f'--{img_names[0]}와 비교 결과--')
    for name, enc in zip(img_names[1:], encodings[1:]):
        embedding = np.linalg.norm(enc-encodings[0], ord=2)
        if embedding < 0.5:
            print(name, '==' ,round(embedding, 4), 'True')
        else:
            print(name, '==' ,round(embedding, 4), 'False')

3.jpg == 0.3975 True
4.jpg == 0.3754 True
7.jpg == 0.4479 True
9.jpg == 0.3811 True
hyo.jpg == 0.5808 False
jo.jpg == 0.6499 False
jung.jpg == 0.6855 False
'''
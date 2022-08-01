import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import imutils
import math
from image_conversion import conversion, video_conversion
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

 

def video_detection():
    url_list, img_list = url_img('faces.txt') # url과 이미지가 담긴 리스트
    
    '''step 1'''
    '''profile image하나씩'''
    for file, img in zip(url_list, img_list):
        profile_image = conversion(img)

        '''now_face 프레임 영상부분 이미지 변형 완료후 사용 사실상 변형완료되면 video로 사용하지 않아도 됌'''
        # # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # with mp_face_mesh.FaceMesh(
        #         max_num_faces=1,
        #         refine_landmarks=True,
        #         min_detection_confidence=0.5,
        #         min_tracking_confidence=0.5) as face_mesh:
        #     while cap.isOpened():
        #         success, image = cap.read()
        #         if not success:
        #             print("웹캠을 찾을 수 없습니다.")
        #             # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
        #             continue

        #         # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        #         image.flags.writeable = False
        #         try:
        #             image = video_conversion(image)
        #         except:
        #             pass
                
        #         result = face_mesh.process(image)
        #         if not result.multi_face_landmarks:
        #             continue
                
        #         height, width = image.shape[:2]



        #         # 이미지 위에 얼굴 그물망 주석을 그립니다.
        #         # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #         if result.multi_face_landmarks:
        #             for facial_landmarks in result.multi_face_landmarks:
                    
        #                 x_min = width
        #                 x_max = 0
        #                 y_min = height
        #                 y_max = 0
                        
        #                 for i in range(0, 468): # 랜드마크 (x, y) 0부터 468  
        #                     pt = facial_landmarks.landmark[i]
        #                     x = int(pt.x * width)
        #                     y = int(pt.y * height)
        #                     if x < x_min:
        #                         x_min = x
        #                     if x > x_max:
        #                         x_max = x
        #                     if y < y_min:
        #                         y_min = y
        #                     if y > y_max:
        #                         y_max = y  
                            
        #                     # cv2.circle(image, (x,y), 1, (250, 5, 0), -1) # 원을 그릴 위치(x,y), 3개의 픽셀, (100, 100, 0)RGB, 점 색상

        #                 # 얼굴 박스
        #                 # cv2.rectangle(image, (x_max,y_min), (x_min,y_max), (250, 50, 0), 1)
                
                
                
        #         # X = [[0, 0], [1, 1]]
        #         # y = [0, 1]
        #         # clf = svm.SVC()
        #         # clf.fit(X, y)
                
        #         # # face_recognition 라이브러리 이용
        #         # now_img_encoding = face_recognition.face_encodings(image)[0]
        #         # profile_img_encoding = face_recognition.face_encodings(profile_image)[0]
        #         # if len(profile_img_encoding) > 0 and len(now_img_encoding) > 0:
        #         #     pass
        #         # else:
        #         #     print('인식안됌')
        #         #     continue
                
        #         # if len(now_img_encoding) > 0 and len(profile_img_encoding) > 0:
        #         #     pass
        #         # else:
        #         #     print('안됌')
        #         #     continue
        #         # result = face_recognition.compare_faces([now_img_encoding], profile_img_encoding)
                
        #         # if result[0] == True:
        #         #     print(file)
        #         #     print(result)
        #         #     cv2.destroyWindow('Image')
                
                
        #         # 보기 편하게 이미지를 좌우 반전
        #         cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(image, 1))
        #         if cv2.waitKey(5) & 0xFF == 27:
        #             break
        # cap.release()
        # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    video_detection()
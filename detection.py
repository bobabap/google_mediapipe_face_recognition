'''
프로필 사진을 url로 받아오고 실시간 비디오로 임베딩 값을 비교한다.

'''
import os
import cv2
import numpy as np
import face_recognition
import dlib
import time
# start = time.time()
# print("time :", time.time() - start)
import urllib.request

import time

model_name='res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='deploy.prototxt.txt'


def detection(url, profile_img_emb):
    
    count = 0
    if count == 20:
        return
    
    cap = cv2.VideoCapture(0)
    
    face = ''
    while True:
        ret, frame = cap.read()
        
        (height, width) = frame.shape[:2]
        model=cv2.dnn.readNetFromCaffe(prototxt_name,model_name)
        blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (300,300),(104.0,177.0,123.0))
        
        model.setInput(blob)
        
        detections=model.forward()
        
        for i in range(0, detections.shape[2]):
        
            confidence = detections[0, 0, i, 2]
            min_confidence=0.5
                
            if confidence > min_confidence:
                
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                if height > endY and width > endX : #예외처리   
                    face = frame[startY:endY,startX:endX]
                    
        cv2.imshow("Frame", frame)
        
        now_face = face_recognition.face_encodings(face)

        # 얼굴인식 안되면 continue
        if now_face == []:
            continue
        
        test = np.linalg.norm(now_face[0] - profile_img_emb, ord=2)
            
        try:
            test
        except:
            return
        
    
        if test < 0.31:
            print(test, '통과')
            print(url)
            return
        else:
            print(test, 'X')
            count = 0
            pass
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return


def run():
    faces_url_list = [] # url리스트
    '''faces.txt 에서 url리스트를 가져온다'''
    with open('faces.txt', "r") as f:
        data = f.readlines()
        for line in data:
            extension = line.split('.')[-1]
            if extension == 'png' or 'jpg': # 확장자 확인
                pass
            else:
                print('Allow png, jpg extensions only')
            
            faces_url_list.append(line.strip())
    
    urlopen = []
    # segmentation fault 방지를 위해 미리 res를 저장하고 사용
    for url in faces_url_list:
        req = urllib.request.Request(url, headers = {"User-Agent" : "Mozilla/5.0"})
        res = urllib.request.urlopen(req).read() # x11\xd3\xdb\xd3\xf2\xab\
        urlopen.append(res)
    
    # res 이미지로 변환
    for res in urlopen:
        image_nparray = np.asarray(bytearray(res), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        
        detection(url, face_recognition.face_encodings(image)[0])
    return
       
    
            
            
if __name__ == '__main__':
    run()
        
        
    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)

    # # 특성별 정규화에 필요한 수치를 계산합니다
    # # (영위상 성분분석 백색화를 적용하는 경우, 표준편차, 평균, 그리고 주성분이 이에 해당합니다)
    # datagen.fit(profile_list)
    # datagen.fit(self_list)
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

from imutils.video import VideoStream
from imutils.video import FPS
import imutils

model_name='res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='deploy.prototxt.txt'


def detection(url, profile_img_emb):
    detector = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    count = 0
    if count == 20:
        return
    
    vs = VideoStream(src=0).start()
    fps = FPS().start()
    face = ''
    while True:
        frame = vs.read()
        frame = imutils.resize(frame,width=600)
        
        (h, w) = frame.shape[:2]
        
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (300,300),(104.0,177.0,123.0), swapRB=False, crop=False)
        
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            
            if confidence > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                
                if fW < 20 or fH < 20:
                    continue
                    
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)

        now_face = face_recognition.face_encodings(frame)

        # 얼굴인식 안되면 continue
        if now_face == []:
            continue
        
        test = np.linalg.norm(now_face[0] - profile_img_emb, ord=2)
        
        try:
            test
        except:
            return
        
        # print(numpy.dot(A, B)/(numpy.linalg.norm(A)*numpy.linalg.norm(B)))

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
        
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()
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
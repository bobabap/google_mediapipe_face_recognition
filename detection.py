'''
find face and crop and embedding

실행 시간 : 0.5425937175750732
'''
import os
import cv2
import numpy as np
import face_recognition
import dlib
import time
# start = time.time()
# print("time :", time.time() - start)
import requests
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

model_name='res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='deploy.prototxt.txt'


def get_face_embedding(img): # img = 이미지 자체
    '''얼굴을 인식하고 임베딩 값을 구한다.'''
    return face_recognition.face_encodings(img)


def crop_embedding(img):

    done_img = ''

    (height, width) = img.shape[:2]
    model=cv2.dnn.readNetFromCaffe(prototxt_name,model_name)
    blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0, (300,300),(104.0,177.0,123.0))

    model.setInput(blob)

    detections=model.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        min_confidence=0.9

        if confidence > min_confidence:

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            if height > endY and width > endX : #예외처리   
                done_img = img[startY:endY,startX:endX]
        else:
            return get_face_embedding(img)

    embedded = get_face_embedding(done_img)
    
    return embedded



def detection(url):
    
    image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    
    protoPath = "deploy.prototxt.txt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    profile_img = crop_embedding(image)

    vs = VideoStream(src=0).start()

    fps = FPS().start()

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
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)

        fps.update()
        cv2.imshow("Frame", frame)
        
        try:
            now_face = get_face_embedding(frame)
            test = np.linalg.norm(now_face[0] - profile_img[0], ord=2)
        except:
            print('fault')
            continue
        
        if test < 0.31:
            print(test, '통과')
            break
        else:
            print(test, 'X')
            pass
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()


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
    
    for url in faces_url_list:
        detection(url)
            
            
if __name__ == '__main__':
    run()
        
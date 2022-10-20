import cv2
import numpy as np
import urllib.request

'''url로 이미지를 받을 경우'''
def url_img(file):
    faces_url_list = [] # url리스트
    '''faces.txt 에서 url리스트를 가져온다'''
    with open(file, "r") as f:
        data = f.readlines()
        for line in data:
            extension = line.split('.')[-1]
            if extension == 'jpg' or 'png' or 'jpeg' or 'jfif': # 확장자 확인
                pass
            else:
                print('Allow png, jpg, jpeg, jfif extensions only')
            
            faces_url_list.append(line.strip())

    '''url list'''
    urlopen = []
    # segmentation fault 방지를 위해 미리 res를 저장하고 사용
    for url in faces_url_list:
        req = urllib.request.Request(url, headers = {"User-Agent" : "Mozilla/5.0"})
        res = urllib.request.urlopen(req).read() # x11\xd3\xdb\xd3\xf2\xab\
        urlopen.append(res)

    '''image list'''
    img_list = []
    # res 이미지로 변환
    for res in urlopen:
        image_nparray = np.asarray(bytearray(res), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        img_list.append(image)

    return faces_url_list, img_list # url, image
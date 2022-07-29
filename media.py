import cv2
import mediapipe as mp
import numpy as np
import urllib.request
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh











land = []
def img_detection():
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

    img_list = []
    # res 이미지로 변환
    for res in urlopen:
        image_nparray = np.asarray(bytearray(res), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        img_list.append(image)
    
    
    # 이미지 파일의 경우을 사용하세요.:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, image in enumerate(img_list):
            
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)

                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            cv2.imwrite(str(idx) + '.png', annotated_image)






def video_detection():
    # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다.")
                # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
                continue

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # 이미지 위에 얼굴 그물망 주석을 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                print(image.shape)
            # 보기 편하게 이미지를 좌우 반전합니다.
            cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    

if __name__ == '__main__':
    # video_detection()
    img_detection()
# google mediapipe [READE.md](README.md)

```markdown
# google mediapipe

```
google_mediapipe_face_rec
├─ detection.py
├─ guide
│  └─ guide_face.png
├─ guide.py
├─ image data
│  ├─ JeongWoo
│  ├─ JiHyo
│  ├─ JungJae
		:
		:
├─ landmark_roc.txt

```
```

### **detection.py**

1. mediapipe 의 face_mesh를 이용해 얼굴을 인식하고 multi_face_landmarks 로 이미지상 얼굴 랜드마크 위치를 얻는다.
2. 얼굴 랜드마크 중 가장 바깥 쪽에 있는 상,하,좌,우 점을 기준으로 +10 -10 하여 얼굴 부분만 자른다.
    
    ```python
    landmark_roc.txt
    
    left_eye_x = int(face_landmarks.landmark[7].x * width) # 왼쪽 눈
    left_eye_y = int(face_landmarks.landmark[7].y * height)
    right_eye_x = int(face_landmarks.landmark[249].x * width) # 오른쪽 눈
    right_eye_y = int(face_landmarks.landmark[249].y * height)
    mid_chin_x = int(face_landmarks.landmark[152].x * width) # 턱끝
    mid_chin_y = int(face_landmarks.landmark[152].y * height)
    										:
    										:
    										:
    ```
    
3. 미간 기준 관자놀이 방향에 따라 얼굴 방향 오른쪽을 보도록 수정

```python
'''Symmetric alignment (face flip)'''
if nose_tip_x < left_temple_x:
    img = cv2.flip(img, 1)
else:
    if right_temple_x < nose_tip_x:
        pass
    else:
        if abs(mid_head_x - left_temple_x) < abs(right_temple_x - mid_head_x): # 왼쪽방향 얼굴 x축 반전
            img = cv2.flip(img, 1)
```

1. 이미지를 다시 processing하여 바뀐 위치의 랜드마크를 얻는다.

```python
'''reset process'''
results = face_mesh.process(img)
if not results.multi_face_landmarks:
    print("worning")
```

1. 턱 끝과 양쪽 관자놀이를 기준으로 수평 변환

```python
'''Image Rotation'''
tan_theta = (mid_chin_x - mid_head_x)/(mid_chin_y - mid_head_y)
theta = np.arctan(tan_theta)
rotate_angle = theta *180/math.pi
image_center = tuple(np.array(img.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D((image_center), -rotate_angle, 1.0)
img  = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))
```

1. guide 사진과의 크기를 맞추기 위해 이미지 크기 조정

```python
img = cv2.resize(img, (300, 300))
```

1. guide 사진에 맞춰 Affine 변환

```
'''Affine Transfomation'''
# [x,y] 좌표점을 4x2의 행렬로 작성
# 좌표점은 좌상->좌하->우상->우하
pts1 = np.float32([[left_temple_x, left_temple_y],[left_cheek_x, left_cheek_y],[right_temple_x, right_temple_y],[right_cheek_x, right_cheek_y]])

# 좌표의 이동점
pts2 = np.float32([[g_left_temple_x, g_left_temple_y],[g_left_cheek_x, g_left_cheek_y],[g_right_temple_x, g_right_temple_y],[g_right_cheek_x, g_right_cheek_y]])

M = cv2.getPerspectiveTransform(pts1, pts2)

img = cv2.warpPerspective(img, M, (200,180), flags=cv2.INTER_LANCZOS4, borderValue=(0,0,0))
```

변환 행렬을 구하기 위해서는 `cv2.getPerspectiveTransform()`
함수가 필요하며, `cv2.warpPerspective()`함수에 변환 행렬 값을 적용하여 최종 결과 이미지를 얻을 수 있다.

### guide.py

guide — guide_face.png

![guide_face](https://user-images.githubusercontent.com/87513112/201913277-6f4d1955-cc32-4cbc-bf07-f156c150684c.png)

Guide 이미지

이 Guide이미지의 Affine변환에 필요한 특정 랜드마크를 구해 반환하여 detection.py에서 사용

### image data

배우들의 얼굴 사진 이미지가 있는 폴더이다.

### 결과

변환 된 이미지를 가지고 유사도를 구하는 기술은 진행 중이다.

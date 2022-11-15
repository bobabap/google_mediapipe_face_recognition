import cv2
import sys
import numpy as np

'''https://blog.naver.com/PostView.naver?blogId=engineerjkk&logNo=222263191204&parentCategoryNo=&categoryNo=30&viewDate=&isShowPopularPosts=true&from=search'''

# 사진 불러오기
src1 = cv2.imread('C:/Users/pc/Desktop/video_detection/crop_img/crop_1.jpg', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('C:/Users/pc/Desktop/video_detection/crop_img/crop_2.jpg', cv2.IMREAD_GRAYSCALE)

#src1 = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)

#src2 = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)


if src1 is None or src2 is None:

    print('Image load failed!')

    sys.exit()


# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)

feature = cv2.KAZE_create()

#feature = cv2.AKAZE_create()

#feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산

kp1, desc1 = feature.detectAndCompute(src1, None)

kp2, desc2 = feature.detectAndCompute(src2, None)

# 특징점 매칭

matcher = cv2.BFMatcher_create()

#matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

matches = matcher.match(desc1, desc2)

# 좋은 매칭 결과 선별

matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:80]

#상위 80개만 선별

print('# of kp1:', len(kp1))

print('# of kp2:', len(kp2))

print('# of matches:', len(matches))

print('# of good_matches:', len(good_matches))

# 호모그래피 계산

pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]

                ).reshape(-1, 1, 2).astype(np.float32)

pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]

                ).reshape(-1, 1, 2).astype(np.float32)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

#good_matches가 dematch의 리스트 80개를 하나하나 받아서 m에 받는다. dematch type에는 queryIndex와 trainIndex가있는데 

#queryIdx는 1번이미지 키포인트 번호이다. 이걸 kp1에대한 인덱스 번호로 준다. 1번영상에서 kpt의 인덱스에 해당하는것을 찾아서 pt라는 점의 좌표를 받아온다. pt는 실수와 좌표를 갖는 두개짜리 튜플이다. 얘들을 ndarray로 받는다. 

#pte1이 N,1,2로 받아야 하기때문에 pts1=에서 reshape함수를 사용해 바꿔준다.

#호모그래피 perspectiveTransform을 H로 받았고, 마스크는 받지않았다.

# 호모그래피를 이용하여 기준 영상 영역 표시

dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None,

                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

(h, w) = src1.shape[:2]

corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]

                    ).reshape(-1, 1, 2).astype(np.float32)

corners2 = cv2.perspectiveTransform(corners1, H)

#perspectiveTransform은 H를 가지고 어디로 이동하는지 좌표계산을 하는것이다. corners1은 N,1,2형태의 shape이어야한다. 

corners2 = corners2 + np.float32([w, 0])

#drawMatches를 호출하면 가로로 붙여서 하나로 만들어주는데 2번영상의 좌표가 1번영상의 가로 크기만큼 쉬프트된다. 그걸 반영하기위해 float32를 더하고 corners에 저장

cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

cv2.namedWindow('dst', cv2.WINDOW_NORMAL)

cv2.imshow('dst', dst)

cv2.waitKey()

cv2.destroyAllWindows()
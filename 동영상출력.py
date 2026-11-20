# 사용 도구 불러오기
import cv2
import matplotlib.pyplot as plt

# YOLO 사용도구 불러오기
from ultralytics import YOLO

# YOLO에서 기본으로 제공하는 모델 불러오기
model = YOLO('yolo11n.pt') # yolo11n : 가벼운 YOLO 모델

# 예측에 사용할 영상 파일 경로 지정
video_path = './data/dog.mp4'

# VideoCapture객체 생성
cap=cv2.VideoCapture(video_path)

# 영상 파일을 이용해서 YOLO모델 사용
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, conf = 0.3)
        # conf파라미터 조절 : 모델이 객체를 탐지할 때
        # 신뢰도를 몇 이상부터 인정할건지를 나타내는 파라미터

        # 검출된 객체에 바운딩박스, 라벨등을 그려서 반환하가
        annotated_frame = results[0].plot()

        # 이미지 표시
        cv2.imshow('YOLO', annotated_frame)

        # 'q'키를 누르면 while 루프 종료
        if cv2.waitKey(1) == ord('q'):
            break
    # 더이상 읽을 프레임 없을 경우에는 루프 종료
    else:
        break

# 영상 파일 닫기
cap.release()

# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
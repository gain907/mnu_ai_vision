import cv2  # 카메라 영상 읽기, 블러/사각형 그리기, 화면 출력용 OpenCV
from ultralytics import YOLO  # YOLO 모델 로드 및 추론용 Ultralytics

# 1. YOLO 모델 로드 (로컬에 없으면 기본 가중치 자동 다운로드)
model = YOLO("yolo11n.pt")

# 2. 웹캠 연결 (0번 카메라 사용, 인식 안 되면 1로 변경해 테스트)
camera = cv2.VideoCapture(0)

# 카운트하고 화면에서 가릴 대상 클래스 지정 (예: "person", "cell phone", "cup" 등)
TARGET_CLASS = "person"

# 3. 실시간 영상 처리 루프
while True:
  # 카메라에서 한 프레임 읽기 (success: 성공 여부, frame: 원본 이미지 배열)
  success, frame = camera.read()
  if not success:
    break  # 프레임을 읽지 못하면 루프 종료

  # 현재 프레임에서 YOLO 객체 탐지 수행 (verbose=False로 콘솔 로그 출력 생략)
  results = model(frame, verbose=False)

  # 첫 번째(현재) 프레임의 탐지 박스 목록 가져오기
  boxes = results[0].boxes

  # 원본 frame을 훼손하지 않기 위해 복사본 생성
  output_frame = frame.copy()

  # 이번 프레임에서 조건에 맞는 대상 객체 개수 초기화
  target_count = 0

  # 탐지된 박스들을 하나씩 순회
  for box in boxes:
    conf = float(box.conf[0])  # 탐지 신뢰도(0.0 ~ 1.0)
    cls_name = results[0].names[int(box.cls[0])]  # 클래스 번호를 클래스 이름 문자열로 변환

    # 신뢰도 0.5 이상이고, 원하는 대상 클래스(TARGET_CLASS)인 경우만 처리
    if conf >= 0.5 and cls_name == TARGET_CLASS:
      target_count += 1  # 대상 객체 카운트 1 증가

      # 바운딩 박스 좌상단(x1, y1) ~ 우하단(x2, y2) 좌표를 정수형으로 변환
      x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())


      # 프레임 해상도(높이 h, 너비 w)를 기준으로 좌표가 화면 밖으로 나가지 않게 보정
      h, w, _ = output_frame.shape
      x1, y1 = max(0, x1), max(0, y1) # x1이나 y1이 0보다 작아 화면 왼쪽·위쪽 바깥으로 나가면 0으로 끌어올립
      x2, y2 = min(w, x2), min(h, y2) # x2가 w를 넘거나 y2가 h를 넘어 화면 오른쪽·아래쪽 바깥으로 나가면 각각 w, h 안쪽으로 맞춥

      # [방법 A] 가우시안 블러로 해당 영역 모자이크 처리 (유효한 영역 너비/높이일 때만 실행)
      if x2 > x1 and y2 > y1:
        roi = output_frame[y1:y2, x1:x2]  # 대상 박스 영역(ROI) 잘라내기
        output_frame[y1:y2, x1:x2] = cv2.GaussianBlur(
            roi, (51, 51), 0
        )  # 커널 (51, 51) 블러 적용

      # [방법 B] 검은색 사각형(-1은 내부 채우기)으로 영역 완전 덮어쓰기 시 아래 줄 주석 해제
      # cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

  # 화면 좌측 상단 (20, 40) 위치에 현재 카운트 텍스트 표시
  cv2.putText(
      output_frame,
      f"{TARGET_CLASS} count: {target_count}",  # 출력할 문자열
      (20, 40),  # 텍스트 시작 좌표 (x, y)
      cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
      1.0,  # 글자 크기 배율
      (0, 255, 0),  # 글자 색상 (BGR 기준 초록색)
      2,  # 글자 선 두께
  )

  # 처리된 결과 프레임을 윈도우 창에 실시간 표시
  cv2.imshow("YOLO Boxes Count & Mask", output_frame)

  # 1ms 동안 키 입력을 대기하며 'q' 키가 눌리면 루프 종료
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# 4. 종료 시 카메라 장치 및 OpenCV 창 리소스 해제
camera.release()
cv2.destroyAllWindows()
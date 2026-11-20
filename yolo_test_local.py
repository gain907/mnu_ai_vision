import cv2  # 영상 캡처 및 화면 출력(imshow)을 위한 OpenCV 라이브러리
from ultralytics import YOLO  # YOLO 모델 로드 및 객체 탐지를 위한 Ultralytics 라이브러리

# 1. YOLO 모델 로드
# 현재 폴더에 'yolo11n.pt'가 있으면 불러오고, 없으면 Ultralytics에서 자동으로 다운로드합니다.
model = YOLO("yolo11n.pt")

# 2. 웹캠 연결
# 0은 기본(내장 또는 첫 번째) 카메라를 의미하며, 인식되지 않으면 1로 바꿔 테스트해 보세요.
camera = cv2.VideoCapture(0)

# 카메라 장치가 정상적으로 열렸는지 확인
if not camera.isOpened():
  print(
      "웹캠을 열 수 없습니다. 카메라 인덱스(0 또는 1)나 macOS 카메라 권한을"
      " 확인하세요."
  )
  exit()  # 카메라를 열 수 없으면 프로그램 종료

print("실행 중... 종료하려면 영상 창을 선택한 상태에서 'q' 키를 누르세요.")

# 3. 실시간 프레임 반복 처리 루프
while True:
  # 카메라에서 한 프레임(이미지 배열)을 읽어옵니다.
  # success: 읽기 성공 여부(True/False), frame: 읽어온 이미지 데이터
  success, frame = camera.read()

  # 프레임을 제대로 못 읽었으면 루프를 중단합니다.
  if not success:
    print("프레임을 읽지 못했습니다.")
    break

  # YOLO 모델로 현재 프레임에서 객체 탐지 수행 (verbose=False로 터미널 로그 출력 억제)
  results = model(frame, verbose=False)

  # 탐지된 바운딩 박스와 클래스 라벨이 그려진 새 이미지를 반환받습니다.
  # plot 화면에 그림만 그림
  # boxes 
  annotated_frame = results[0].plot()

  # 4. 로컬 창에 실시간 영상 출력
  # 첫 번째 인자는 창 제목, 두 번째 인자는 화면에 띄울 이미지입니다.
  cv2.imshow("YOLO Local Webcam Detection", annotated_frame)

  # 1ms 동안 키 입력을 대기하며, 'q' 키가 눌리면 루프를 종료합니다.
  # 0xFF 마스킹은 64비트 OS 환경에서 키 코드 하위 8비트만 정확히 비교하기 위해 사용합니다.
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# 5. 프로그램 종료 전 카메라 및 창 리소스 정리
camera.release()  # 웹캠 점유 해제
cv2.destroyAllWindows()  # 열려 있는 모든 OpenCV 창 닫기
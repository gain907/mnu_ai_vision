import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
camera = cv2.VideoCapture(0)

while True:
  success, frame = camera.read()
  if not success:
    break

  results = model(frame, verbose=False)
  boxes = results[0].boxes  # 현재 프레임의 탐지 박스 모음

  for box in boxes:
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    cls_name = results[0].names[cls_id]

    # 신뢰도 0.5 이상인 객체만 처리
    if conf >= 0.5:
      x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
      print(f"[{cls_name}] conf={conf:.2f}, box=({x1}, {y1}, {x2}, {y2})")

      # 예시: 'person'이 탐지되면 해당 영역만 잘라내기(crop)
      if cls_name == "person":
        cropped = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        # 필요한 경우 cropped를 저장하거나 별도 모델/처리로 넘길 수 있습니다.

  annotated_frame = results[0].plot()
  cv2.imshow("YOLO Local Webcam Detection", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()
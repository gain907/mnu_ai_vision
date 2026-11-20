from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# YOLO 모델
model = YOLO("yolo11n.pt")

# 웹캠
camera = cv2.VideoCapture(0)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO 실시간 탐지</title>
</head>
<body>
    <h2>YOLO 실시간 탐지</h2>
    <img src="/video" width="800">
</body>
</html>
"""


def generate():
    while True:
        success, frame = camera.read()

        if not success:
            break

        # YOLO 탐지
        results = model(frame, verbose=False)

        # 탐지 결과 그리기
        frame = results[0].plot()

        # JPEG 변환
        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


app.run(host="0.0.0.0", port=5001)
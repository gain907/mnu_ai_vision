# 사용 도구 불러오기
import cv2
import matplotlib.pyplot as plt

# cv2.waitKey(delay) 함수 : 프레임간의 간격을 제어하고 사용자의 입력을 처리할 수 있는 기능
# 주어진 시간(delay)동안 사용자의 키입력을 기다려서 사용자의 입력이 들어올때까지 대기하며, 해당 키의 아스키 코드를 반환
# 매개변수 delay는 대기시간을 나타내며 단위는 밀리초단위 -> 30을 적을 경우 30밀리초 (1000ms -> 1s)
# 0을 입력할 경우에는 무한대기를 진행


# 동영상 파일을 열기 위해서는 VideoCapture 객체를 생성해야한다.
# VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

# 영상 파일 작업실시
# 영상 파일이 열려있는 동안 반복할 코드
while cap.isOpened():
    # cap.read() : 한 프레임씩 일거오기
    success, frame = cap.read()
    # success : 프레임 읽기에 성공했는지의 여부 (True / False)
    # frame : 읽어들인 영상 플레임 (이미지)의 정보

    # 프레임 읽기에 성공한 경우
    if success :
        # 프레임을 표시
        cv2.imshow('camera', frame)

        # 조건을 하나 걸기 : 'q'버튼을 누를 경우에 while 루프를 종료하여 영상 재생을 중지하겠다.
        if cv2.waitKey(1) == ord('q'):   #1 =  0.001초(1ms), 0 은 정지화면 대기용
            print('사용자에 의해서 영상을 종료했습니다.')
            break
    # 영상 객체가 이미지를 읽을 수 없을 경우 while루프를 종료하기 (과부화 방지)
    else:
        break

# VideoCapture 객체 종료
cap.release()
# 비디오 파일 또는 카메라 장치 등 외부 자원을 사용하는데 사용이 끝난 경우에 종료를 하지 않으면
# 계속해서 자원을 사용하고 있기 때문에 우리가 사용할 자원이 줄어든다. 그리고 이게 계속 반복될
# 경우에는 과부하가 올 수 있다.
# 그렇기 때문에 꼭 사용을 했으면 종료를 해줘야 된다.

#모든 창을 닫아주기
cv2.destroyAllWindows()
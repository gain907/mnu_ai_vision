# 사용 도구 불러오기
import cv2
import matplotlib.pyplot as plt

# cv2.waitKey(delay) 함수 : 프레임간의 간격을 제어하고 사용자의 입력을 처리할 수 있는 기능
# 주어진 시간(delay)동안 사용자의 키입력을 기다려서 사용자의 입력이 들어올때까지 대기하며, 해당 키의 아스키 코드를 반환
# 매개변수 delay는 대기시간을 나타내며 단위는 밀리초단위 -> 30을 적을 경우 30밀리초 (1000ms -> 1s)
# 0을 입력할 경우에는 무한대기를 진행


# cv2를 사용해서 이미지를 출력해보자
# 이미지 로드
img = cv2.imread('./data/pet.jpg')
# 이미지 출력
cv2.imshow('pets', img)
# 무한대기
cv2.waitKey(5000)
# 모든 openCV창 종료
cv2.destroyAllWindows()
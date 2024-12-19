import cv2
from ultralytics import YOLO

# 1. 훈련된 YOLO 모델 로드
model = YOLO("C:/strawberry_model/weights/best.pt")  # 훈련된 모델 경로 지정

# 2. 영상 로드
video_path = "C:/code_list/strawberry.mp4"  # 분석할 영상 파일 경로
cap = cv2.VideoCapture(video_path)

# 출력 영상 설정
output_path = video_path.replace('.mp4', '_output.mp4')  # 저장할 파일 경로
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 영상 FPS 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 영상 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 영상 높이

# VideoWriter 객체 생성 (코덱: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 3. 영상 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO 모델로 추론
    results = model(frame,conf=0.005)  # 프레임 전달
    
    # 결과 그리기
    annotated_frame = results[0].plot()  # 결과를 시각화한 프레임 생성
    
    # 화면에 표시
    cv2.imshow("Detection", annotated_frame)

    # 저장 (시각화된 프레임을 출력 파일에 저장)
    out.write(annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 자원 해제
cap.release()
cv2.destroyAllWindows()

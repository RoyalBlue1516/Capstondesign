import os
import shutil
from ultralytics import YOLO

def train_model(data_path, model_save_path):
    """
    YOLO 모델 학습
    """
    print("모델 학습 시작...")
    model = YOLO("yolov8n.yaml")  # YOLOv8 기본 모델
    results = model.train(data=data_path, epochs=100, imgsz=640)
    print(f"모델 학습 완료. 결과는 {results.save_dir} 디렉터리에 저장됨.")

    # 최적의 모델 파일(best.pt) 복사
    best_model_path = os.path.join(results.save_dir, "best.pt")
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, model_save_path)
        print(f"최적의 모델이 {model_save_path}에 저장되었습니다.")
    else:
        print(f"최적의 모델 파일을 찾을 수 없습니다: {best_model_path}")


if __name__ == "__main__":
    # 경로 설정
    base_path = os.path.abspath("./바운딩박스작업")  # 코드와 데이터의 루트 폴더
    dataset_path = os.path.join(base_path, "miner_dataset")  # 증강된 데이터셋 경로
    yaml_path = os.path.join(dataset_path, "dataset.yaml")  # YAML 파일 경로
    model_save_path = os.path.join(base_path, "yolov8_model.pt")  # 학습된 모델 저장 경로

    # 학습 실행
    train_model(yaml_path, model_save_path)

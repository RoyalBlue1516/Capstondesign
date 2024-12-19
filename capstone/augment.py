import os
import cv2
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO

# 증강 파이프라인 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 좌우 반전
    A.VerticalFlip(p=0.5),    # 상하 반전
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),  # 이동, 확대/축소, 회전
    A.HueSaturationValue(p=0.5), # 색상, 채도, 명도 랜덤 조정
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),     # 밝기 및 대비 랜덤 조정
    A.GaussNoise(var_limit=(10, 50), p=0.5),  # 가우시안 노이즈 추가
    A.Resize(416, 416),  # YOLO 모델 정규화
],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),  # 바운딩박스를 YOLO 형식으로 지정
)

def filter_valid_bboxes(bboxes, category_ids):
    """
    유효한 바운딩 박스 필터링 (YOLO 형식 좌표 값이 (0, 1] 범위)
    """
    valid_bboxes = []
    valid_category_ids = []
    for bbox, category_id in zip(bboxes, category_ids):
        if all(0 < coord <= 1 for coord in bbox[:4]):  # YOLO 좌표 범위 확인
            valid_bboxes.append(bbox)
            valid_category_ids.append(category_id)
    return valid_bboxes, valid_category_ids


def augment_and_save(image_path, label_path, output_dir, transform, prefix="aug_"):
    """
    증강된 이미지와 라벨 데이터를 저장
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"오류: 이미지를 로드할 수 없습니다. 경로: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 라벨 로드
    with open(label_path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        category_ids = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # 클래스 ID
            bbox = [float(x) for x in parts[1:]]

            # 유효성 검사 (YOLO 형식 바운딩 박스 검증)
            if len(bbox) == 4:
                bboxes.append(bbox)
                category_ids.append(class_id)

    # 유효한 바운딩 박스 필터링
    bboxes, category_ids = filter_valid_bboxes(bboxes, category_ids)

    if not bboxes:  # 유효한 바운딩 박스가 없으면 건너뜀
        print(f"경고: {label_path}에서 유효한 바운딩 박스를 찾지 못했습니다.")
        return

    # 데이터 증강 수행
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_category_ids = transformed['category_ids']

    # 저장 경로 생성
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, prefix + os.path.basename(image_path))
    output_label_path = os.path.join(output_dir, prefix + os.path.splitext(os.path.basename(label_path))[0] + '.txt')

    # 증강된 이미지 저장
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, transformed_image)

    # 증강된 라벨 저장
    with open(output_label_path, 'w') as f:
        for bbox, category_id in zip(transformed_bboxes, transformed_category_ids):
            f.write(f"{category_id} {' '.join(map(str, bbox))}\n")
            
def prepare_augmented_dataset(image_path, label_path, output_dir, transform, augmentations=5):
    """
    이미지와 라벨 데이터를 증강하여 새로운 폴더에 저장
    """
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for image_file in os.listdir(image_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_full_path = os.path.join(image_path, image_file)
            label_full_path = os.path.join(label_path, os.path.splitext(image_file)[0] + ".txt")

            if os.path.exists(label_full_path):
                for i in range(augmentations):  # 지정된 횟수만큼 증강
                    prefix = f"aug_{i}_"
                    augment_and_save(image_full_path, label_full_path, output_dir, transform, prefix=prefix)

    print(f"증강 데이터가 {output_dir}에 저장되었습니다.")
    return output_dir


# 바운딩 박스 처리된 이미지 출력
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
height, width =416,416
category_id_to_name = {0: 'miner', 1: 'strawberry_mite'}

# 이미지에 바운딩박스 시각화
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness) 
	
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img
 
def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


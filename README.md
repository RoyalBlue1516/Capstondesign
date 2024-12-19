# 데이터 증강 프로그램 사용 방법

## 환경 설정
1. Python 환경에서 필수 라이브러리를 설치합니다:
   ```bash
   pip install albumentations opencv-python ultralytics matplotlib
   ```
2. YOLO 형식의 바운딩 박스 라벨 데이터(.txt)와 이미지 데이터를 준비합니다.

---

## 프로그램 사용 방법

### 1. 데이터 증강 실행
- 아래 코드를 사용하여 증강 데이터를 생성합니다:

```python
from augment import prepare_augmented_dataset, transform

prepare_augmented_dataset(
    image_path="path/to/images",         # 원본 이미지 폴더 경로
    label_path="path/to/labels",         # 원본 라벨 폴더 경로
    output_dir="path/to/output",         # 증강 결과 저장 경로
    transform=transform,                  # 증강 파이프라인 설정
    augmentations=5                       # 증강 횟수
)
```

- **결과 저장 구조:**
  - 증강된 이미지: `output/images/` 디렉터리에 저장.
  - 증강된 라벨: `output/labels/` 디렉터리에 YOLO 형식(.txt)으로 저장.

---

### 2. 증강 결과 시각화
- 증강된 이미지와 바운딩 박스를 확인하려면 아래 코드를 사용합니다:

```python
from augment import visualize

visualize(
    image=transformed_image,        # 증강된 이미지 데이터
    bboxes=transformed_bboxes,      # 바운딩 박스 좌표
    category_ids=transformed_category_ids,  # 클래스 ID
    category_id_to_name={0: 'class_1', 1: 'class_2'}  # 클래스 ID와 이름 매핑
)
```
- 출력: 바운딩 박스와 클래스 이름이 표시된 이미지가 화면에 표시됩니다.

---

### 3. YOLO 모델 학습 준비
- 증강된 데이터 세트를 YOLOv8 모델 학습용으로 준비합니다.
  - YOLO 형식의 데이터 경로를 모델 학습 코드에서 참조합니다:
    ```yaml
    path: path/to/output  # 증강된 데이터 폴더 경로
    train: images/train   # 학습 데이터 경로
    val: images/val       # 검증 데이터 경로
    ````

---

## 프로그램 구조

### 주요 함수
| 함수명                     | 설명                                             |
|----------------------------|--------------------------------------------------|
| `prepare_augmented_dataset` | 이미지와 라벨 데이터를 증강하여 저장합니다.        |
| `visualize`                 | 증강된 이미지와 바운딩 박스를 시각화합니다.        |

### 주요 라이브러리
- **Albumentations**: 데이터 증강
- **OpenCV**: 이미지 처리
- **Ultralytics**: YOLOv8 모델 학습 지원
- **Matplotlib**: 이미지 시각화

---

## 실행 예시
```bash
python augment.py
```
- 실행 후 `path/to/output` 디렉터리에서 증강된 데이터를 확인할 수 있습니다.

---


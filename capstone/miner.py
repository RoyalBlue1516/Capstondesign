import os
from augment import prepare_augmented_dataset, transform

# 상대 경로 설정
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, 'miner_dataset', 'images')
label_path = os.path.join(current_dir, 'miner_dataset', 'labels')
output_dir = os.path.join(current_dir, 'miner_dataset', 'augmented')

# 데이터 증강 수행
if __name__ == "__main__":
    prepare_augmented_dataset(image_path, label_path, output_dir, transform, augmentations=5)
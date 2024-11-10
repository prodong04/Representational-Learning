from datasets import load_dataset

# 데이터셋 로드
ds = load_dataset("mertcobanov/animals")

# 학습 세트와 테스트 세트로 분할 (60:40 비율)
train_test_split = ds['train'].train_test_split(test_size=0.4, seed=42)

# 테스트 세트를 다시 검증 세트와 테스트 세트로 분할 (50:50 비율)
val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

# 최종 데이터셋 구성
train_dataset = train_test_split['train']
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# 테스트 세트에서 'image' 열만 남기기
test_images = test_dataset.remove_columns([col for col in test_dataset.column_names if col != 'image'])

# 테스트 세트에서 'label' 열만 남기기
ground_truth = test_dataset.remove_columns([col for col in test_dataset.column_names if col != 'label'])

# 각 데이터셋을 로컬 디스크에 저장
train_dataset.save_to_disk("dataset/train")
val_dataset.save_to_disk("dataset/val")
test_images.save_to_disk("dataset/test_images")
ground_truth.save_to_disk("dataset/ground_truth")

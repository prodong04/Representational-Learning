import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# 디바이스 설정 (GPU 사용 가능 시 GPU로 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 학습 데이터셋 로드
train_dataset = load_from_disk("dataset/train")

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋에 전처리 적용
def transform_examples(batch):
    batch['pixel_values'] = [preprocess(image.convert("RGB")) for image in batch['image']]
    return batch

# 'image' 필드를 제거하기 위해 remove_columns 사용
train_dataset = train_dataset.map(transform_examples, batched=True, remove_columns=['image'])

# 배치 크기 설정
batch_size = 32

# 데이터 로더 생성
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 사전 학습된 ResNet-50 모델 로드
model = models.resnet50(pretrained=True)

# 출력 레이어 수정 (데이터셋의 클래스 수에 맞게 설정)
num_classes = len(train_dataset.features['label'].names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 모델을 디바이스로 이동
model = model.to(device)

# 모델 평가 함수 정의
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# 모델 평가 및 정확도 계산
all_labels, all_preds = evaluate_model(model, train_loader, device)
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 예측 확률 수집 함수 정의
def collect_predicted_probabilities(model, data_loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch['pixel_values'].to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)

# 예측 확률 수집
all_probs = collect_predicted_probabilities(model, train_loader, device)

# ECE 계산 함수 정의
def compute_ece(pred_probs, true_labels, num_bins=10):
    confidences = np.max(pred_probs, axis=1)
    predictions = np.argmax(pred_probs, axis=1)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]

        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == true_labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

# ECE 계산
ece = compute_ece(all_probs, np.array(all_labels))
print(f"ECE: {ece:.4f}")

# 추론용
# 클래스화 및 스레드 지원 추가

# torchvision 관련 라이브러리들을 import
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageFile


# 이미지 분류기 클래스
class ImageClassifier:
    # 생성자
    def __init__(self, model_path='./model-pretrained.pth', label_path='./img_classes.txt'):
        # 이미지 로드 시 잘린 이미지에 대한 경고 미출력 설정
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # 디바이스 설정 (cuda:0 혹은 cpu 사용)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 모델과 라벨 파일 경로 설정
        self.model_path = model_path
        self.label_path = label_path

        # 모델과 라벨 초기화
        self.labels = self._load_labels()
        self.model = self._load_model()
        

    # 모델 로드
    def _load_model(self):
        # ResNet101 모델 초기화
        model = models.resnet101(weights="DEFAULT")

        # 분류를 위한 Fully Connected Layer 정의
        fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.labels))
        )

        # 모델의 Fully Connected Layer를 새로 정의한 fc로 교체
        model.fc = fc

        # 모델을 지정한 디바이스로 이동
        model.to(self.device)

        # CUDA가 사용 가능한 경우 모델 가중치 로드
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(self.model_path))
        else:
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        # 모델을 평가 모드로 설정
        model.eval()
        return model

    # 라벨 로드
    def _load_labels(self):
        # 라벨 파일을 읽어와 리스트로 반환
        with open(self.label_path) as file:
            labels = [line.strip() for line in file.readlines()]
        return labels

    # 이미지 전처리
    def preprocess_image(self, img):
        # 이미지 전처리를 위한 변환 연산 정의
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 이미지에 전처리 적용 및 디바이스로 이동
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t.to(self.device)

    # 이미지 분류
    def predict(self, filepath):
        # 딕셔너리 생성
        result_dict = {}

        # 파일 경로가 비어있는 경우 빈 결과 반환
        if len(filepath) == 0:
            return result_dict

        # 이미지 파일 열기
        img = Image.open(filepath)
        
        # 이미지 전처리 수행
        img_tensor = self.preprocess_image(img)

        # 모델에 이미지 전달하여 예측 수행
        with torch.no_grad():
            out = self.model(img_tensor)

        # 최대값 및 퍼센티지 계산
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        self.labels[index[0]], percentage[index[0]].item()

        # 예측 결과 출력
        #print("분류 결과 : {0}, {1}".format(self.labels[index[0]], percentage[index[0]].item()))

        # 예측 결과 및 퍼센티지를 딕셔너리에 저장
        #print("---" * 12)
        _, indices = torch.sort(out, descending=True)
        for idx in indices[0][:len(self.labels)]:
            #print("{0}){1}, {2}".format(idx, self.labels[idx], percentage[idx].item()))
            result_dict[self.labels[idx]] = percentage[idx].item()
        #print("---" * 12)

        # 딕셔너리 반환
        return result_dict

# 클래스 사용 예시
if __name__ == "__main__":
    # ImageClassifier 클래스 인스턴스 생성
    classifier = ImageClassifier()

    # 이미지 파일에 대한 예측 수행
    result = classifier.predict('test1.jpg')
    print(result)

    # 이미지 파일에 대한 예측 수행
    result = classifier.predict('test2.jpg')
    print(result)
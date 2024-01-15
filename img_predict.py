# 추론용
# 클래스화 및 스레드 지원(?) 필요

# torchvision 관련 라이브러리들을 import
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageFile


# 잘린 이미지 Load 시 경고 미출력
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 1234

# 개별 이미지 판별
def predict_set(filepath):
    # device 설정 (cuda:0 혹은 cpu 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 결과 {운동기구 : 가중치} 형태임
    result_dict = {}

    #파일명 비어있을 경우
    if(len(filepath)==0):
        return
    img = Image.open(filepath)
    
    # 라벨 불러오기
    with open('./img_classes.txt') as file:
        labels = [line.strip() for line in file.readlines()]
    
    model = models.resnet101(weights="DEFAULT") 
    fc = nn.Sequential(
        nn.Linear(2048, 256), # 모델의 features의 출력이 7X7, 512장 이기 때문에 in_features=7*7*512 로 설정 함
        nn.ReLU(), 
        nn.Linear(256, 64), 
        nn.ReLU(), 
        nn.Linear(64, 3), # 현재 3개 클래스 분류이기 때문에 3로 out_features=3로 설정 함
    )
    model.fc = fc
    model.to(device)
    
    # cuda 여부에 따라 다르게 load
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./model-pretrained.pth'))
    else:
        model.load_state_dict(torch.load('./model-pretrained.pth', map_location=torch.device('cpu')))

    model.eval()
    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 전처리 실행
    img_t = preprocess(img)

    # unsqueeze
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t.to(device))

    # 최대값 뽑기
    _, index = torch.max(out, 1)

    # 라벨과 퍼센티지
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    labels[index[0]], percentage[index[0]].item()

    print("분류 결과 : {0}, {1}".format(labels[index[0]],percentage[index[0]].item()))

    _, indices = torch.sort(out, descending=True)
    print("---"*12)
    for idx in indices[0][:len(labels)]:
        print("{0}){1}, {2}".format(idx, labels[idx], percentage[idx].item()))
        result_dict[labels[idx]] = percentage[idx].item()
    print("---"*12)

    return result_dict
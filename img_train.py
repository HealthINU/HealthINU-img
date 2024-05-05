# 학습용

# torchvision 관련 라이브러리들을 import
import torch
from tqdm import tqdm
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import random
from PIL import Image, UnidentifiedImageError,ImageFile

import glob

# 잘린 이미지 Load 시 경고 미출력
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 1234

# 폴더 생성 함수
def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)

# 이미지 정규성 검증
def validate_image(filepath):
    try:
        # PIL.Image로 이미지 데이터를 로드
        img = Image.open(filepath).convert('RGB') 
        img.load()
    except UnidentifiedImageError: # corrupt 된 이미지는 해당 에러 출력
        print(f'오염된 이미지 오류: {filepath}')
        return False
    except (IOError, OSError): # Truncated (잘린) 이미지에 대한 에러 출력
        print(f'잘린 이미지 오류: {filepath}')
        return False
    else:
        return True

  
# 시드 고정 함수
def fix_seed():
    # PyTorch의 랜덤시드 고정
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0) # gpu가 1개 이상일 경우

    # Numpy 랜덤시드 고정
    np.random.seed(0)

    # CuDNN 랜덤시드 고정
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True # 연산 처리 속도가 줄어들어서 보통 연구 후반기에 사용함

    # 파이썬 랜덤시드 고정
    random.seed(SEED)


# 학습
def train_set(filepath, isColab=False, set_epochs=30, set_lr=0.0001, batch_size = 32, workers=8, early_patience=0, optim_name="Adam", isFreeze=True):
    # 로그 폴더 생성
    if not os.path.exists('log'):
        os.makedirs('log')

    # 로그 파일 생성
    from datetime import datetime 
    s = datetime.now().strftime("log/%Y%m%d_%H%M%S_train_log.txt") 
    print(s) # 20XX0XXX_XX27_train_log.txt 의 이름으로 저장됨

    # device 설정 (cuda:0 혹은 cpu 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # image 데이터셋 root 폴더
    root = filepath
    dirs = os.listdir(root)

    # 시드 고정
    fix_seed()

    # 코랩일 경우 /로 경로가 나뉘고, 일반적인 경우 \\로 경로가 나뉨
    if(isColab): # True
        token1 = '/'
        token2 = '/*'
    else: # False
        token1 = '\\'
        token2 = '\\*'
        
    # 폴더 별로 이미지 데이터셋을 살펴보고, 오염된 이미지를 제거
    for dir_ in dirs:
        folder_path = os.path.join(root, dir_)
        files = os.listdir(folder_path)
        images = [os.path.join(folder_path, f) for f in files]
        for img in images:
            valid = validate_image(img)
            if not valid:
                # corrupted 된 이미지 제거함
                os.remove(img)
    
    # 폴더 경로를 리스트로 저장
    folders = glob.glob(root+token2)

    print(folders)
    print("%d folders found".format(len(folders)))

    with open(s, "a") as file:
        file.write("-------------------------------------- \n")
        for folder in folders:
            file.write(folder)
            file.write("\n")
        file.write("--------------------------------------\n")
    

    # train: test ratio. 0.2로 설정시 test set의 비율은 20%로 설정됨
    test_size = 0.2

    # train / test 셋의 파일을 나눔
    train_images = []
    test_images = []

    # 텍스트 파일 저장용
    labels_txt = ""

    # 폴더 별로 for문
    for folder in folders:
        label = os.path.basename(folder)
        labels_txt += label + "\n"


        files = sorted(glob.glob(folder + token2))

        # 각 라벨마다 이미지 데이터셋 셔플
        random.shuffle(files)

        idx = int(len(files) * test_size)
        train = files[:-idx]
        test = files[-idx:]

        train_images.extend(train)
        test_images.extend(test)

    # 라벨 저장
    with open('./img_classes.txt','w+') as file:
        file.write(labels_txt)

    # train, test 전체 이미지 셔플
    random.shuffle(train_images)
    random.shuffle(test_images)

    # Class to Index 생성
    class_to_idx = {os.path.basename(f):idx for idx, f in enumerate(folders)}
    print(class_to_idx)

    # Label 생성
    # 마지막에서 두 번째 요소를 라벨로 사용
    train_labels = [f.split(token1)[-2] for f in train_images]
    test_labels = [f.split(token1)[-2] for f in test_images]


    # 데이터셋 정보 출력
    print('==='*12)
    print(f'[Dataset INFO]')
    print(f'Train images: {len(train_images)}')
    print(f'Train labels: {len(train_labels)}')
    print(f'Test images: {len(test_images)}')
    print(f'Test labels: {len(test_labels)}')
    print('==='*12)

    # 이미지 전처리
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),          # 이미지 리사이즈
        transforms.CenterCrop((224, 224)),      # 중앙 Crop
        transforms.RandomHorizontalFlip(0.5),   # 50% 확률로 Horizontal Flip
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),      
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # train, test 데이터셋 생성
    train_dataset = CustomDataset(train_images, train_labels, class_to_idx, train_transform)
    test_dataset = CustomDataset(test_images, test_labels, class_to_idx, test_transform)

    # train, test 데이터 로더 생성 => 모델 학습시 입력하는 데이터셋
    # 총 횟수는 이미지수 / 배치 사이즈
    # 배치 사이즈의 기본 값은 32로 설정, 테스트시 변경 가능
    # num_workers는 코랩일 경우 2, 아닐 경우 8로 설정
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=workers)

    with open(s, "a") as file:
        file.write("Batch Size : {} \n".format(batch_size))
    
    # ResNet 모델 생성
    # Convolutional layer - "필터"는 이미지를 통과하여 한 번에 몇 Pixel(NxN)을 스캔하고 각 형상이 속하는 클래스를 예측하는 형상 맵을 만듦
    model_set = "resnet101"

    if(model_set=="resnet101"):
        model = models.resnet101(weights="DEFAULT") 
    elif(model_set=="resnet50"):
        model = models.resnet50(weights="DEFAULT") 

    with open(s, "a") as file:
        file.write("Model : {} \n".format(model_set))

    # 가중치를 Freeze 하여 학습시 업데이트가 일어나지 않도록 설정
    if(isFreeze):
        for param in model.parameters():
            param.requires_grad = False  # 가중치 Freeze

    # Fully-Connected Layer를 Sequential로 생성하여 pretrained 모델의 'Classifier'에 연결
    # Convolution/Pooling 네트워크 프로세스의 최종 결과를 취해서 
    # 분류 결정에 도달하는 완전히 연결된 계층(Fully-Connected Layer)
    # 2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층
    fc = nn.Sequential(
        nn.Linear(2048, 256), # 모델의 features의 출력이 1X1, 2048장 이기 때문에 in_features=1*1*2048 로 설정
        nn.ReLU(), 
        nn.Linear(256, 64), 
        nn.ReLU(), 
        nn.Linear(64, len(folders)), # 폴더 갯수만큼 클래스 수 맞춤
    )
    model.fc = fc
    model.to(device)

    # 옵티마이저를 정의 
    # 옵티마이저에는 model.parameters()를 지정해야 함
    # 학습률 (learning rate)은 기본은 0.0001로 설정
    lr=set_lr
    if(optim_name == "Adam"): optimizer = optim.Adam(model.parameters(), lr)
    elif(optim_name == "SGD"): optimizer = optim.SGD(model.parameters(), lr)
    else: optimizer = optim.Adam(model.parameters(), lr) # 기본값은 Adam

    with open(s, "a") as file:
        file.write("Optimizer : {} \n".format(optim_name))

    # 손실함수(loss function)을 지정
    # Multi-Class Classification 이기 때문에 CrossEntropy 손실을 지정함
    loss_fn = nn.CrossEntropyLoss()

    # 최대 Epoch을 지정
    num_epochs = set_epochs
    model_name = 'model-pretrained'

    with open(s, "a") as file:
        file.write("Learning Rate : {} \n".format(lr))
        file.write("Total Epochs : {} \n".format(num_epochs))
        file.write("--------------------------------------\n")

    # 최소 loss를 inf으로 설정
    min_loss = np.inf

    # Early Stopping을 위한 변수를 0으로 초기화
    early_cnt = 0

    # Epoch 별 훈련 및 검증을 수행 함
    for epoch in range(num_epochs):
        # 모델 훈련

        # 훈련 손실과 정확도를 반환
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)

        # 검증 손실과 검증 정확도를 반환
        val_loss, val_acc = model_eval(model, test_loader, loss_fn, device)   
            
        # val_loss가 개선되었다면 min_loss를 갱신 후 model의 가중치(weights)를 저장
        if val_loss < min_loss:
            # 개선되었으므로 early_cnt를 0으로 초기화
            early_cnt = 0
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')

            # 로그 파일에도 저장
            with open(s, "a") as file:
                file.write(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!\n')
            min_loss = val_loss
            torch.save(model.state_dict(), f'{model_name}.pth')
            
        # Epoch 별 결과를 출력
        print(f'Epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

        # Epoch 별 결과를 로그 파일에 저장
        with open(s, "a") as file:
                file.write(f'Epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}\n')

        # Early Stopping을 위한 조건문
        # val_loss가 개선되지 않으면 early_cnt를 증가시킴
        if (val_loss >= min_loss & early_patience > 0):
            #early stopping 카운트
            early_cnt += 1
            print(f'[INFO] EarlyStopping counter: {early_cnt} out of {early_patience}')

            # 로그 파일에 저장
            with open(s, "a") as file:
                file.write(f'[INFO] EarlyStopping counter: {early_cnt} out of {early_patience}\n')
            
            # 만약 이 상태에서 early_cnt가 early_patience와 같거나 커진 상태라면 학습을 종료
            if early_cnt >= early_patience:
                print(f'[INFO] Early stopping at {epoch+1} epoch')

                # 로그 파일에 저장
                with open(s, "a") as file:
                    file.write(f'[INFO] Early Stopping at {epoch+1} epoch\n')
                break
    
    # 모델에 저장한 가중치를 로드
    model.load_state_dict(torch.load(f'{model_name}.pth'))

    # 최종 검증 손실(validation loss)와 검증 정확도(validation accuracy)를 산출
    final_loss, final_acc = model_eval(model, test_loader, loss_fn, device)
    print(f'Evaluation loss: {final_loss:.5f}, Evaluation accuracy: {final_acc:.5f}')
    # 결과를 로그 파일에 저장
    with open(s, "a") as file:
        file.write(f'Evaluation loss: {final_loss:.5f}, Evaluation accuracy: {final_acc:.5f}\n')


class CustomDataset(Dataset): 
    # 파일 경로, 라벨, class_to_idx, transform을 입력으로 받음
    def __init__(self, files, labels, class_to_idx, transform):
        super(CustomDataset, self).__init__()
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    # 길이 출력
    def __len__(self): 
        return len(self.files)
    
    # 데이터셋의 idx번째 데이터를 반환
    def __getitem__(self, idx):
        # file 경로
        file = self.files[idx]
        # PIL.Image로 이미지 로드
        img = Image.open(file).convert('RGB')
        # transform 적용
        img = self.transform(img) 
        # label 생성
        lbl = self.class_to_idx[self.labels[idx]] 
        # image, label 반환
        return img, lbl


def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정, training mode 일 때 Gradient 가 업데이트 됨, 반드시 train()으로 모드 변경을 해야 함
    model.train()
    
    # loss와 accuracy 계산을 위한 임시 변수임 0으로 초기화됨
    running_loss = 0
    corr = 0
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑
    prograss_bar = tqdm(data_loader)
    
    # mini-batch 학습을 시작
    for img, lbl in prograss_bar:
        # image, label 데이터를 device에 올림
        img, lbl = img.to(device), lbl.to(device)
        
        # 누적 Gradient를 초기화함
        optimizer.zero_grad()
        
        # Forward Propagation을 진행하여 결과 얻기
        output = model(img)
        
        # 손실함수에 output, label 값을 대입하여 손실을 계산함
        loss = loss_fn(output, lbl)
        
        # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산함
        loss.backward()
        
        # 계산된 Gradient를 업데이트함
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환함
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출함
        _, pred = output.max(dim=1)
        
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산 item()은 tensor에서 값을 추출함
        # 합계는 corr 변수에 누적함
        corr += pred.eq(lbl).sum().item()
        
        # loss 값은 1개 배치의 평균 손실(loss), img.size(0)은 배치사이즈(batch size)
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됨
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출함
        running_loss += loss.item() * img.size(0)
        
    # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됨
    acc = corr / len(data_loader.dataset)
    
    # 평균 손실(loss)과 정확도를 반환
    # train_loss, train_acc
    return running_loss / len(data_loader.dataset), acc


def model_eval(model, data_loader, loss_fn, device):
    # model.eval()은 모델을 평가모드로 설정을 바꿈
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차임
    model.eval()
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 0으로 초기화
        corr = 0
        running_loss = 0
        
        # 배치별 evaluation을 진행
        for img, lbl in data_loader:
            # device에 데이터를 올림
            img, lbl = img.to(device), lbl.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출
            output = model(img)
            
            # output의 max(dim=1)은 max probability와 max index를 반환
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출
            _, pred = output.max(dim=1)
            
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산함, item()은 tensor에서 값을 추출함
            # 합계는 corr 변수에 누적함.
            corr += torch.sum(pred.eq(lbl)).item()
            
            # loss 값은 1개 배치의 평균 손실(loss), img.size(0)은 배치사이즈(batch size) 
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됨
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출함
            running_loss += loss_fn(output, lbl).item() * img.size(0)
        
        # validation 정확도를 계산함
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출함
        acc = corr / len(data_loader.dataset)
        
        # 결과를 반환함
        # val_loss, val_acc
        return running_loss / len(data_loader.dataset), acc
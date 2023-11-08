# 학습과 추론용

# torchvision 관련 라이브러리들을 import
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image, UnidentifiedImageError,ImageFile

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
        print(f'Corrupted Image is found at: {filepath}')
        return False
    except (IOError, OSError): # Truncated (잘린) 이미지에 대한 에러 출력
        print(f'Truncated Image is found at: {filepath}')
        return False
    else:
        return True
    
def predict_set(filepath):
    print("분류")


def train_set(filepath):
    print("학습")


class CustomDataset(Dataset): 
  def __init__(self): #데이터셋의 전처리를 해주는 부분
     a=1
  def __len__(self): #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
     a=1
  def __getitem__(self, idx): #데이터셋에서 특정 1개의 샘플을 가져오는 함수
     a=1



#train_dataset = datasets.STL10('/train', split='train', download=True, transform=transforms.ToTensor())
#test_dataset = datasets.STL10('/test', split='test', download=True, transform=transforms.ToTensor())
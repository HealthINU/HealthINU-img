# 학습과 추론용

# torchvision 관련 라이브러리들을 import
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# 폴더 생성 함수
def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)

class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self): #데이터셋의 전처리를 해주는 부분
     a=1
  def __len__(self): #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
     a=1
  def __getitem__(self, idx): #데이터셋에서 특정 1개의 샘플을 가져오는 함수
     a=1


#train_dataset = datasets.STL10('/train', split='train', download=True, transform=transforms.ToTensor())
#test_dataset = datasets.STL10('/test', split='test', download=True, transform=transforms.ToTensor())
# HealthINU-img
## 개요
### Resnet101을 기반 전이학습을 통한 운동기구 이미지 분류기 (졸업 프로젝트중 일부)
### 운동기구 이미지를 종류별로 분류하는 코드 (학습 코드 포함)
### 사용 데이터셋은 직접 크롤링을 통해 수집 후 전처리를 거침
#### [데이터셋 다운로드 링크](https://www.dropbox.com/scl/fi/jevweydnkpjc5egc0hccq/HealthINU-img-dataset.zip?rlkey=cm3wud0b62ck6qg5fdh6mfdqs&st=l5u5rleq&dl=0)
##### (문제시 연락 좀)

## Overview
### Exercise Equipment Image Classifier Using Transfer Learning Based on ResNet101 (Part of Graduation Project)
### A code for classifying exercise equipment images by type (including training code)
### The dataset was collected through web crawling and preprocessed before use
#### [Dataset DL Link](https://www.dropbox.com/scl/fi/jevweydnkpjc5egc0hccq/HealthINU-img-dataset.zip?rlkey=cm3wud0b62ck6qg5fdh6mfdqs&st=l5u5rleq&dl=0)
##### (Call me if there's a problem with this dataset)

## 기본 호출법
```
커맨드라인에서 호출
img_main.py [이미지 파일 이름] >> 개별 이미지 파일 분류 (모델 필요)
img_main.py !train >> 학습 모드 (모델 생성함)
[이 경우 dataset 내 폴더들의 이름이 라벨이 되고 그 안 이미지들이 학습 대상]
img_main.py !test >> 테스트 파일들 돌려서 결과 알려줌 (모델 필요)
ipynb 파일도 있는데 학습 시 이거 사용해도 무방
```

## Basic Usage
```
Run from the command line:  
img_main.py [image file name]  >> Classifies a single image file (requires a trained model)  
img_main.py !train  >> Training mode (creates a new trained model)
[In this case, folder names within the dataset serve as labels, and the images inside them are used for training.]  
img_main.py !test  >> Runs test files and outputs results (requires a trained model)  
An IPython Notebook (.ipynb) file is also available and can be used for training.  
```

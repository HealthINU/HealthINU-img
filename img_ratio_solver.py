# 이미지의 종횡비를 1:1로 맞추는 코드, 다만 비율이 1:1이 아닌 경우에는 비율을 유지하되, 검은색 배경을 추가하여 1:1로 맞춤

import os
from PIL import Image, ImageOps
from tqdm import tqdm
from pathlib import Path

def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)

def img_ratio_solver(dir_trans, dir_back, dir_output='./output', isWhite=False):
    # 출력 폴더 생성하기
    createFolder(dir_output)

    # 이미지 목록 생성하기
    # images1: 이미지 목록
    images1 = Path(dir_trans).glob('*.jpg')

    # 이미지 목록을 리스트로 변환하기
    images1 = list(images1)

    # 둘중 하나라도 비어있는 경우
    if len(images1)==0:
        print("There are no images in the folder.")
        return -1

    # 이미지 배치 생성
    for i in tqdm(range(len(images1))):
        # 이미지 불러오기
        img1 = Image.open(images1[i]).convert("RGB")

        # 이미지 정리
        img_merger(img1, dir_output + f'/{i+1}.jpg', isWhite)
    
    print("Image ratio solving complete.")

def img_merger(img1, dir_output, isWhite):
    # img1: 운동기구 이미지

    # img1의 높이가 너비보다 큰지 비교
    # (사실 랫 풀 다운 같은거는 높이가 너비보다 커가지고)
    if img1.size[1] > img1.size[0]:
        # 크다면 img1의 높이 길이로 패딩 추가
        if(isWhite):
            img1 = ImageOps.pad(img1, (img1.size[1], img1.size[1]), color=(255, 255, 255))
        else:
            img1 = ImageOps.pad(img1, (img1.size[1], img1.size[1]), color=(0, 0, 0))

    # img1의 너비가 더 큰경우
    elif img1.size[0] > img1.size[1]:
        # 크다면 img1의 너비 길이로 패딩 추가
        if(isWhite):
            img1 = ImageOps.pad(img1, (img1.size[0], img1.size[0]), color=(255, 255, 255))
        else:
            img1 = ImageOps.pad(img1, (img1.size[0], img1.size[0]), color=(0, 0, 0))
    
    rgb_im = img1.convert('RGB')
    rgb_im.save(dir_output)


def main():
    #   ./crawled/@최종정리용/new_bench_press_machine/white/transparent
    dir_trans= input('Enter Transparent Image Folder Path : ')
    dir_back = "./bg" #input('Enter Background Image Folder Path : ')
    dir_output_name = input('Enter Output Folder name : ')
    dir_output = "./dataset_ratio_white/"+dir_output_name
    isWhite = True
    img_ratio_solver(dir_trans, dir_back, dir_output, isWhite)


if __name__ == "__main__":
	main()

# 이미지 부족 사태 대비용 이미지 배치 복제기
# 배경 제거된 이미지를 헬스장 배경 이미지 위에 랜덤하게 배치하여 새로운 이미지 배치 생성

# A 폴더의 투명 이미지들과 B 폴더의 배경 이미지들이 있을 때 
# A 폴더의 이미지 순서대로 고른 하나,
# B 폴더의 이미지 중 랜덤으로 하나를 골라 위에 랜덤하게 배치하여 새로운 이미지를 생성하는게 아이디어임

import os
from PIL import Image, ImageFilter
from tqdm import tqdm
from pathlib import Path

def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)

def img_batch_duplicator(dir_trans, dir_back, dir_output='./output'):
    # 출력 폴더 생성하기
    createFolder(dir_output)

    # 이미지 목록 생성하기
    # images1: 투명 이미지 목록, images2: 배경 이미지 목록
    images1 = Path(dir_trans).glob('*.png')
    images2 = Path(dir_back).glob('*.jpg')

    # 이미지 목록을 리스트로 변환하기
    images1 = list(images1)
    images2 = list(images2)

    # 둘중 하나라도 비어있는 경우
    if len(images1)==0 or len(images2)==0:
        print("There are no images in the folder.")
        return -1

    # 이미지 배치 생성
    for i in tqdm(range(len(images1))):
        # 이미지 불러오기
        img1 = Image.open(images1[i]).convert("RGBA")
        img2 = Image.open(images2[i%len(images2)])

        # 이미지 머지
        img_merger(img1, img2, dir_output + f'/{i+1}.jpg')
    
    print("Image batch duplication complete.")
    

def gaussian_blur(img, radius=2):
    # 가우시안 블러 적용
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def img_merger(img1, img2, dir_output):
    # img1: 투명 운동기구 이미지, img2: 배경 이미지

    # img1의 높이가 너비보다 큰지 비교
    # (사실 랫풀다운용임)
    if img1.size[1] > img1.size[0] * 1.2:
        # 크다면 임시 투명 이미지 생성
        # 사이즈는 높이와 너비가 img1의 높이 길이와 같은 이미지 ( 1대1 비율로 맞추기 위함 )
        img_temp = Image.new('RGBA', (img1.size[1], img1.size[1]), (255, 255, 255, 0))
        img_temp.paste(img1, (0, 0), img1)
        img1 = img_temp
    # img1의 너비가 더 큰경우
    elif img1.size[0] > img1.size[1] * 1.2:
        # img1의 중앙을 높이와 같은 길이로 자르기 ( 1대1 비율로 맞추기 위함 )
        left = (img1.size[0] - img1.size[1]) / 2
        right = (img1.size[0] + img1.size[1]) / 2
        img1 = img1.crop((left, 0, right, img1.size[1]))
        

    # img2를 img1의 크기로 조정
    img2 = img2.resize(img1.size)
    
    # img2에 가우시안 블러 적용
    img2 = gaussian_blur(img2, 3)

    # 실제로 이미지 머지
    img2.paste(img1, (0, 0), img1)
    rgb_im = img2.convert('RGB')
    rgb_im.save(dir_output)


def main():
    #   ./crawled/@최종정리용/new_bench_press_machine/white/transparent
    dir_trans= input('Enter Transparent Image Folder Path : ')
    dir_back = "./bg" #input('Enter Background Image Folder Path : ')
    dir_output_name = input('Enter Output Folder name : ')
    dir_output = "./output/"+dir_output_name
    img_batch_duplicator(dir_trans, dir_back, dir_output)


if __name__ == "__main__":
	main()

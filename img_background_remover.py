# 폴더 내 이미지 파일 배경 투명 처리
import os
from time import sleep
from tqdm import tqdm
from pathlib import Path
from rembg import remove, new_session
def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)

def img_remove(path):
    session = new_session()
    files = Path(path).glob('*.jpg')
    # 폴더 내 jpg들
    for file in tqdm(files):
        input_path = str(file)
        output_path = str(file.parent / "transparent" / (file.stem + "_out.png"))
        
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)


def main():
    path = input("경로 입력 : ")
    # transparent 폴더 생성
    dir = path + "\\transparent\\"
    createFolder(dir)

    # 경로 예제
    # ./crawled/@최종정리용/new_bench_press_machine/white
    img_remove(path)

# main 함수 로딩부
if __name__ == '__main__':
    main()

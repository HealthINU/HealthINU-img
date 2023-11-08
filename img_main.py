# 메인맨
import sys

# 학습함수, 테스트함수 불러오기
from img_train_predict import train_set, predict_set, validate_image

def main():
    # 인수 있는지 확인
    try:
        print(sys.argv[1])
    except:
        print("매개변수가 없?는듯???")
        input("아무 키나 입력해서 종료 : ")
        exit()
    
    # 학습인 경우
    if(sys.argv[1]=="!train!"):
         print("이미지 학습 시작")

    # 이미지 오염 여부 확인
    elif(validate_image(sys.argv[1])):
        print("이미지 분류 시작")
        #predict_set(sys.argv[1]) # 아직구현안됨
    else: print("이미지가 손상됨")

    input("아무 키나 입력해서 종료 : ")
    exit()

if __name__ == "__main__":
	main()
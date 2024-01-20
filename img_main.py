# 메인맨
import sys

# 학습함수, 테스트함수 불러오기
from img_train import train_set, validate_image
from img_predict import ImageClassifier

def main():
    # 인수 있는지 확인
    try:
        print(sys.argv[1])
    except:
        print("인수 없음")
        exit()

    # 학습인 경우
    if(sys.argv[1]=="!train"):
         print("이미지 학습 시작")
         train_set('.\\dataset')

    elif(sys.argv[1]=="!train"):
         print("모델 정보 출력")
         

         
    # 학습이 아닌 경우
    else:
        # 이미지 오염 여부 확인
        if(validate_image(sys.argv[1])):
            print("이미지 분류 시작")
            # ImageClassifier 클래스 인스턴스 생성
            classifier = ImageClassifier()

            # 이미지 파일에 대한 예측 수행
            result = classifier.predict(sys.argv[1])
            print(result)
        else: print("이미지 문제 발생")

    print("종료")
    exit()

if __name__ == "__main__":
	main()
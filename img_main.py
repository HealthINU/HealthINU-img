# 메인맨
import sys
import json

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

    # 테스트인 경우
    elif(sys.argv[1]=="!test"):
        # ImageClassifier 클래스 인스턴스 생성
        classifier = ImageClassifier()

        correct=0
        wrong=0

        # test_answer.json 파일 불러오기
        with open('test_answer.json') as f:
            test_answer = json.load(f)

        # json에 대한 반복문 수행
        for key in test_answer:
            # 이미지 검증
            if not(validate_image(key['name'])):
                print("이미지 문제 발생")
                continue

            # 해당 파일에 대한 예측 수행
            result = classifier.predict(key['name'])

            # 예측 결과 출력
            print("---" * 12)
            print("[", key['name'], "]")
            print("예측 : ", list(result.keys())[0], ", 정답 : ", key['answer'])
            if(key['answer'] == list(result.keys())[0]):
                print("결과 : 일치")
                correct+=1
            else:
                print("결과 : 불일치!!")
                wrong+=1
            print("---" * 12)
        
        # 정답률 및 개수 출력
        print("---" * 12   )
        print("전체 개수 : ", correct + wrong, "개")
        print("정답 : ", correct, "개")
        print("오답 : ", wrong, "개")
        print("정답률 : ", correct / (correct+wrong) * 100, "%")
        print("---" * 12   )
         
         
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
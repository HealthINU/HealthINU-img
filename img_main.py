# 메인맨
import sys
import json
import os

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
        # 학습 파라미터 설정
        # epochs: 학습 횟수, learning_rate: 학습률, batch_size: 배치 크기, num_workers: 데이터 로드에 사용할 프로세스 수
        epochs            = 50
        learning_rate     = 0.0001
        batch_size        = 32
        num_workers       = 2

        # early_patience: 조기 종료를 위한 기다리는 횟수, optimizer: 최적화 알고리즘, weight_decay: 가중치 감쇠
        early_patience    = 15
        optimizer         = "Adam" # "Adam" or "AdamW" or "RAdam" or "SGD
        weight_decay      = 0

        # isFixedSeed: 시드 고정 여부, isFreeze: 미세 조정을 위한 레이어 고정 여부
        isFixedSeed       = False
        isFreeze          = False 

        print("이미지 학습 시작")

        # 2번째꺼는 코랩이 아니라서 False
        train_set(".\\dataset", False, epochs, learning_rate, batch_size, num_workers, early_patience, weight_decay, optimizer, isFixedSeed, isFreeze)

        # 학습 후 현재시간 가져옴
        from datetime import datetime 
        s = datetime.now().strftime("models/model-pretrained_%Y%m%d_%H%M%S.pth")

        # 현재 model-pretrained.pth 모델을 복사해서 s로 된 이름으로 저장
        if not os.path.exists("models"):
            os.makedirs("models")
        import shutil
        shutil.copy("model-pretrained.pth", s)
        print("Model saved")

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
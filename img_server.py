# 메인맨
import sys
import torch

# 파이썬 flask
from flask import Flask, request
from flask_restx import Api, Resource

# 이미지 검증 함수 및 테스트 함수 불러오기
from img_train import validate_image
from img_predict import ImageClassifier

# API 서버를 구축하기 위한 기본 구조
app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
api = Api(app)

# ImageClassifier 클래스 인스턴스 생성
classifier = ImageClassifier()
print("Model loaded successfully!")

# 이미지 처리
@api.route('/process')
class ProcessImage(Resource):
    # POST 요청 처리
    def post(self):
        data = request.get_json()

        # 이미지 경로가 없는 경우
        if 'image' not in data: 
            return {"error": "No image path provided"}, 400
        image_path = data['image']

        # 이미지 경로가 비어있는 경우
        if image_path == '': 
            return {"error": "No path in image"}, 400
        
        # 이미지 경로가 올바른 경우
        if image_path:
            # 이미지 검증 성공 시
            if validate_image(image_path): 
                print("이미지 분류 시작")
                # 이미지 파일에 대한 분류 수행
                result_dict = classifier.predict(image_path)
                top3_result_dict = []

                # 상위 3개 결과만 반환
                for k in list(result_dict.keys())[:3]:
                    top3_result_dict.append({"name" : k, "prob": result_dict[k]})
                    
                # JSON 형식으로 변환해서 전송
                return {"message": "Image processed successfully!","result": top3_result_dict}, 200 
            # 이미지 검증 실패 시
            else:
                return {"error": "Invalid image"}, 400
        # 이미지 경로가 올바르지 않은 경우
        else:
            return {"error": "Invalid path"}, 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
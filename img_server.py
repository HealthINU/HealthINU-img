# 메인맨
import sys
import torch

# 파이썬 flask
from flask import Flask, request
from flask_restx import Api, Resource

# 학습함수, 테스트함수 불러오기
from img_train_predict import train_set, predict_set, validate_image

# API 서버를 구축하기 위한 기본 구조
app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
api = Api(app)

# 이미지 처리
@api.route('/process')
class ProcessImage(Resource):
    def post(self):
        data = request.get_json()
        if 'image' not in data: 
            return {"error": "No image path provided"}, 400
        image_path = data['image']
        if image_path == '': 
            return {"error": "No path in image"}, 400
        if image_path:
            if validate_image(image_path): 
                print("이미지 분류 시작")
                result_dict = predict_set(image_path)
                top3_result_dict = {k: result_dict[k] for k in list(result_dict.keys())[:3]} # 상위 3개 
                return {"message": "Image processed successfully!","result": top3_result_dict}, 200 # JSON 형식으로 변환해서 전송
            else:
                return {"error": "Invalid image"}, 400
        else:
            return {"error": "Invalid path"}, 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
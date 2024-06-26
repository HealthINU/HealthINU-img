# 이미지 크롤링용
# img_main.py 과는 독립적으로 작동함
# 모델 저장 및 불러오기용

from tqdm import tqdm
import os
import urllib.request
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def createFolder(dir):
    # 생성되지 않은 경우에만 생성
    if not os.path.exists(dir):
        os.makedirs(dir)


def findImages(keyword, url, url_mod):
    # URL 모드가 아닐경우
    # URL 설정
    if not url_mod:
        url = 'https://www.google.com/imghp'
    
    # 저장할 폴더 생성
    dir = ".\crawled"
    createFolder(dir)
    dir += "\\" + keyword + "\\"
    createFolder(dir)
    
    # 로그 텍스트 파일 생성
    with open(dir + keyword + ".txt", 'w') as f:
        from datetime import datetime 
        f.write(datetime.now().strftime("%Y%m%d_%H%M%S \n") )
        f.write(keyword + "\n")
        f.write(url)

    # 크롬 드라이버 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('headless') # 이거 쓰면 headless 됨
    chrome_options.add_argument('window-size=1920x1080')
    chrome_options.add_argument('log-level=3')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(time_to_wait=10)

    # URL 모드가 아닐경우
    # 검색어를 검색 후 엔터
    if(not url_mod):
        keyElement = driver.find_element(By.NAME, "q")
        keyElement.send_keys(keyword)
        keyElement.send_keys(Keys.RETURN)

    # 많은 이미지를 구하기 위해 스크롤 내려야 함
    SCROLL_PAUSE_TIME = 1
    # 스크롤 높이 측정
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 스크롤 아래로 내림
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 새로운 페이지가 load되기를 기다림
        time.sleep(SCROLL_PAUSE_TIME)

        # 새로운 스크롤 높이 구하여 이전 스크롤 높이와 비교
        new_height = driver.execute_script("return document.body.scrollHeight")
        try:
            if new_height == last_height: # 끝 도달 시
                driver.find_element_by_css_selector(".mye4qd").click() # 더보기 버튼 클릭
        except: # 더보기 버튼이 없을 경우
            break # 탈출
        last_height = new_height
    
    # 이미지 리스트
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    images = driver.find_elements(
        By.XPATH,
        "//img[contains(@class, 'YQ4gaf') and not(contains(@class, 'zr758c'))]",  # Change it yourself, filter out small pictures, like pictures, etc.
    )

    #images = driver.find_elements(By.CSS_SELECTOR, ".YQ4gaf")
    #images = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i.Q4LuWd')
    print(f"Found {len(images)} images")

    # images가 비어있을 경우 위 코드를 다시 실행
    if not images:
        print("No images found, retrying...")
        driver.close()
        time.sleep(0.5)
        return 1
    
    # 이미지 마다 반복
    prograss_bar = tqdm(images)
    count = 1
    
    for image in prograss_bar:
        '''if(count % 2 ==0): # 임시 조치 (2개씩 받는 이슈 해결 필요)
            image.click()
            time.sleep(1)
            count = count + 1
            continue'''
        try:
            image.click()
            time.sleep(2.5)

            # 이미지 URL 추출 (src)
            imgUrl = driver.find_element(
                By.XPATH,
                '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]'
            ).get_attribute("src")

            # 오프너로 열기
            opener = urllib.request.build_opener()
            opener.addheaders = [
                ('User-Agent',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
            ]
            urllib.request.install_opener(opener)

            # 이미지 파일 저장
            urllib.request.urlretrieve(imgUrl, f'{dir}{keyword}_{str(count)}.jpg')
            #print(imgUrl)
            opener.close()
            count = count + 1

        except Exception as e:
            # 암호화된 이미지는 무시
            # print('Error : ', e)
            pass
    
    print('Download Complete')
    # 크롬 드라이버 종료
    driver.close()
    return 0


def main():
    keyword = input('Enter Image Folder Name : ')
    url_mod = True
    url = input("Enter your url : ")
    result = 1
    # 성공할 때까지 자동 재시도
    while result != 0:
        result = findImages(keyword, url, url_mod)
    exit()

if __name__ == "__main__":
	main()
# 이미지 크롤링용
# img_main.py 과는 독립적으로 작동함
# 모델 저장 및 불러오기용

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


def findImages(keyword, url_mod):
    # URL 설정
    url = 'https://www.google.com/imghp'

    # URL 모드일경우 덮어 씌우게 하기
    if(url_mod):
        url = input("Enter your url : ")
    
    # 저장할 폴더 생성
    dir = ".\crawled"
    createFolder(dir)
    dir += "\\" + keyword + "\\"
    createFolder(dir)

    # 크롬 드라이버 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    #chrome_options.add_argument('headless') # 이거 쓰면 headless 됨
    chrome_options.add_argument('window-size=1920x1080')
    chrome_options.add_argument("disable-gpu")
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
        if new_height == last_height:
            try:
                driver.find_element_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height
    
    # 이미지 리스트
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    # 이미지 마다 반복
    count = 1
    for image in images:
        try:
            image.click()
            time.sleep(2.5)

            # 이미지 URL 추출 (src)
            imgUrl = driver.find_element(
                By.XPATH,
                '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img'
            ).get_attribute("src")

            # 오프너로 열기
            opener = urllib.request.build_opener()
            opener.addheaders = [
                ('User-Agent',
                'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')
            ]
            urllib.request.install_opener(opener)

            # 이미지 파일 저장
            urllib.request.urlretrieve(imgUrl, f'{dir}{keyword}_{str(count)}.jpg')
            #print(imgUrl)

            if (count % 50==0): print('Downloaded {} images'.format(count))
            count = count + 1
        except Exception as e:
            # 암호화된 이미지는 무시
            # print('Error : ', e)
            pass
    
    print('Download Complete')
    # 크롬 드라이버 종료
    driver.close()


def main():
    keyword = input('Enter Image Folder Name : ')
    url_mod = True
    findImages(keyword, url_mod)
    exit()

if __name__ == "__main__":
	main()
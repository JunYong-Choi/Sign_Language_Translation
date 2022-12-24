import cv2
import sys, os
import time
import mediapipe as mp
from modules.utils import createDirectory
import numpy as np
from PIL import ImageFont, ImageDraw, Image

fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

createDirectory('dataset')

# 손가락 통증 이슈로 나누어 찍기
# actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# actions = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
# actions = ['ㅗ', 'ㅛ', 'ㅜ']
actions = ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
# seq_length = 10
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())


# 열렸는지 확인
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 웹캠의 속성 값을 받아오기
# 정수 형태로 변환하기 위해 round
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

if fps != 0:
    delay = round(1000/fps)
else:
    delay = round(1000/30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')


# 프레임을 받아와서 저장하기
while cap.isOpened():
    for idx, action in enumerate(actions):
        
        os.makedirs(f'dataset/output_video/{action}', exist_ok=True)

        videoFolderPath = f'dataset/output_video/{action}'
        videoList = sorted(os.listdir(videoFolderPath), key=lambda x:int(x[x.find("_")+1:x.find(".")]))
      
        if len(videoList) == 0:
            take = 1
        else:
            f = videoList[-1].find("_")
            e = videoList[-1].find(".")
            take = int(videoList[-1][f+1:e]) + 1

        saved_video_path = f'dataset/output_video/{action}/{action}_{take}.avi'

        out = cv2.VideoWriter(saved_video_path, fourcc, fps, (w, h))

        ret, img = cap.read()
        if not ret:
            break
         
        # 한글 폰트 출력    
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{action.upper()} 입력 대기중..', font=font, fill=(255, 255, 255))
        img = np.array(img_pil)

        #cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(4000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                break
            
            # 비디오 녹화
            out.write(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)

            # esc를 누르면 강제 종료
            if cv2.waitKey(delay) == 27: 
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
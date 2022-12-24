import cv2
import sys, os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

videoFolderPath = "dataset/output_video"
videoTestList = os.listdir(videoFolderPath)

testTargetList =[]

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)

print("---------- Start Video List ----------")
testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)
print(testTargetList)
print("----------  End Video List  ----------\n")

for target in testTargetList:
    print("Now Streaming :", target)
    cap = cv2.VideoCapture(target)

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

    # 프레임을 받아와서 저장하기
    while True:
        ret, img = cap.read()

        if not ret:
            break

        # draw box
        cv2.rectangle(img, (0,0), (w, 70), (245, 117, 16), -1)

        # draw text target name
        #cv2.putText(img, target, (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 한글 폰트 출력    
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((15,20), target[23:], font=font, fill=(0, 0, 0))
        img = np.array(img_pil)


        cv2.imshow('img', img)
        cv2.waitKey(delay)

        # esc를 누르면 강제 종료
        if cv2.waitKey(delay) == 27: 
            break


    cap.release()
    cv2.destroyAllWindows()
print("\n---------- Finish Video Streaming ----------")
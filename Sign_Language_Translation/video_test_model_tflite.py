import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import math
import os
from PIL import ImageFont, ImageDraw, Image
from modules.utils import Vector_Normalization
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)


actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
             'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10

# MediaPipe holistic model
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test video path
videoFolderPath = "dataset/example1"
videoTestList = os.listdir(videoFolderPath)

testTargetList =[]

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)

testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)

for target in testTargetList:

    cap = cv2.VideoCapture(target)

    seq = []
    action_seq = []
    last_action = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            
            # 오른손 랜드마크 리스트
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]


            # 벡터 정규화
            vector, angle_label = Vector_Normalization(joint)

            # 위치 종속성을 가지는 데이터 저장
            # d = np.concatenate([joint.flatten(), angle_label])
        
            # 정규화 벡터를 활용한 위치 종속성 제거
            d = np.concatenate([vector.flatten(), angle_label.flatten()])

            seq.append(d)

            if len(seq) < seq_length:
                continue

            # Test model on random input data.
            # input_shape = input_details[0]['shape']
            # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            
            # 시퀀스 데이터와 넘파이화
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_data = np.array(input_data, dtype=np.float32)

            # tflite 모델을 활용한 예측
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            y_pred = interpreter.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
            conf = y_pred[0][i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

                if last_action != this_action:
                    last_action = this_action
            
            # 한글 폰트 출력    
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            # org=(int(right_hand_lmList.landmark[0].x * img.shape[1]), int(right_hand_lmList.landmark[0].y * img.shape[0] + 20))
            draw.text((10,30), f'{action.upper()}', font=font, fill=(255, 255, 255))
            
            img = np.array(img_pil)


        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break


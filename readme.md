## ***실시간 수어 번역 인식 모듈생성 (with. MediaPipe, LSTM)***

***

- 실시간 영상 및 녹화 동영상을 활용해 지문자 인식 프로그램 생성

- 활용방안
    - 수어 사용자를 위한 학습 보조 프로그램 개발 가능
    - 실시간 수어 사용자와의 의사소통 프로그램 개발 가능

- 수어종류
    - 자모음(31개에 대한 한글 자모 지문자)

![%EC%A7%80%EB%AC%B8%EC%9E%90_%EC%9D%B4%EB%AF%B8%EC%A7%80.jpg](attachment:%EC%A7%80%EB%AC%B8%EC%9E%90_%EC%9D%B4%EB%AF%B8%EC%A7%80.jpg)

출처 : https://www.urimal.org/1222

***

### ***데이터 수집***

- 31개의 자음, 모음에 대한 팀원 3명의 학습영상 촬영

![%ED%8C%80%EC%9B%903%EB%AA%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%83%9D%EC%84%B1.gif](attachment:%ED%8C%80%EC%9B%903%EB%AA%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%83%9D%EC%84%B1.gif)

각각 'ㅎ' , 'ㅏ' , 'ㄱ' 에 대한 data 생성중...

***

### ***데이터 전처리***

![hand_landmarks.png](attachment:hand_landmarks.png)

출처 : https://google.github.io/mediapipe/solutions/hands.html

수집한 영상 데이터에서 위와 같이 각 hand keypoint의 Vector, Angle 값을 인식해 데이터로 사용하게 됩니다.

본 프로젝트에서는 지문자를 활용하기 때문에 한손 keypoint만 활용하였습니다.

***

### ***Pipeline***

- making_video.py
    - 원하는 자,모음을 설정해 동영상을 생성합니다. (openCV 활용)
    
- create_dataset_from_video.py
    - video data를 사용하여 hand keypoint의 Vector, Angle 값을 sequence data로 변환해 npy 파일로 저장합니다.
    
- train_hand_gesture.ipynb
    - npy file load하여 모델을 생성합니다.
    
- video_test_model_tflite.py
    - videoFolderPath를 지정하여 저장된 비디오를 활용하여 테스트합니다.
    
- webcam_test_model_tflite.py
    - webcam을 활용하여 실시간으로 테스트합니다.

***

### ***시연***

#### Using webcam

![KakaoTalk_20221210_214325078.gif](attachment:KakaoTalk_20221210_214325078.gif)

***

### ***결론***

저작권 문제로 여기엔 넣지 못했지만 타 youtube 동영상에 대입시켜본 결과 정확히 지문자를 분류 하는 성능을 보였다.  
추가적으로 왼쪽 위에 뜨는 문자들을 결합해 text data까지 생성시킬 수 있도록 project를 향후 발전 시킬 계획에 있다.
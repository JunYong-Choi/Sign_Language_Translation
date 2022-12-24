### User Guide [예제 동작 순서]

1. making_video.py
- 원하는 양 손 제스쳐 동영상을 촬영합니다.
- 동영상 촬영은 30초를 기준으로 하고 중단을 원하면, "ESC" 키를 눌러 종료합니다.
- 촬영을 여러번 진행할 수 있습니다.
- [주의] 촬영 중 인식되는 랜드마크는 동영상에 저장되지 않습니다.

2. show_video.py (생략 가능)
- 촬영한 데이터를 확인합니다.
- video_path를 지정하면, 경로의 하위 모든 비디오 파일을 재생합니다.

3. create_dataset_from_video.py
- 촬영한 데이터를 활용하여 양손 관절 및 각도를 시퀀스 데이터로 변환하여 npy 파일로 저장합니다.

4. train_hand_gesture.ipynb
- npy file load하여 모델을 생성합니다.
- keras model, tflite model 두 종류의 모델을 생성합니다. (tflite model만 활용)

5. video_test_model : videoFolderPath를 지정하여 비디오를 활용하여 테스트합니다.
- tflite.py : tflite model을 테스트합니다.

6. webcam_test_model : 웹캠을 활용하여 테스트합니다.
- tflite.py : tflite model을 테스트합니다.

c.f) unicode.py는 자음모음 결합을 위해 사용하려 했으나 사용 x


<br>

### 데이터 정규화 방법 선택
- 필요에 따라 데이터 정규화를 선택 적용합니다. (적용 or 적용 X)
- 2가지 코드 중 1개만 선택합니다.

```python
# 위치 종속성을 가지는 데이터 저장
d = np.concatenate([joint.flatten(), angle_label])

# 정규화 벡터를 활용한 위치 종속성 제거
d = np.concatenate([vector.flatten(), angle_label.flatten()])

```
(Examples) 벡터 정규화를 적용하는 코드
```python
# 위치 종속성을 가지는 데이터 저장
# d = np.concatenate([joint.flatten(), angle_label])

# 정규화 벡터를 활용한 위치 종속성 제거
d = np.concatenate([vector.flatten(), angle_label.flatten()])

```
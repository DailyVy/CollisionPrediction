# AISolution

0. Model
 Yolo v8, Depth Anythings v2, Seggpt 를 사용
해당 모델에 사용되는 requirements 설치 필요

2. checkpoints
- Depth Anythings : checkpoints폴더 안에 Depth Anythings v2 (https://github.com/DepthAnything/Depth-Anything-V2) 에서 제공하는 Pre-trained Model 저장
- Seggpt : https://github.com/baaivision/Painter/tree/main/SegGPT/SegGPT_inference 에서 Pre-trained Model을 받아 저장

2. examples
- Seggpt을 위한 이미지 및 Mask

3. python
- seggpt_v20_basic.py 파일 실행 후 index.html 접속 시 영상 수신 가능
- '결과 화면'을 선택 시 사람 인식 마스크와 호이스트 위치가 표시
- 호이스트 동작감지 체크 후 저장 시, Optical Flow 표시

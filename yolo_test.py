import argparse
import cv2
import os
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# YOLO 모델을 이용한 객체 탐지 및 시각화 함수
def detect_and_annotate(yolo_model, image):
    # YOLO 모델로 객체 탐지 수행
    results = yolo_model(image, conf=0.2, iou=0.5, device=DEVICE)

    # Annotator 생성
    annotator = Annotator(image, line_width=2, font_size=1, font='Arial.ttf')

    # 탐지된 객체들에 대해 바운딩 박스와 레이블 그리기
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            c = int(detection.cls)
            conf = float(detection.conf)
            label = f'{result.names[c]} {conf:.2f}'
            annotator.box_label(detection.xyxy.squeeze(), label, color=colors(c, True))

    return annotator.im

# 입력 경로에 따라 이미지 불러오기
def load_images(input_path):
    if os.path.isdir(input_path):
        # 폴더의 모든 이미지 파일 불러오기
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    elif os.path.isfile(input_path):
        # 단일 이미지 파일 경로
        image_paths = [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    return image_paths

# 메인 함수
def main(args):
    # YOLO 모델 로드
    yolo_model = YOLO(args.yolo_model)

    # 입력 이미지 불러오기
    image_paths = load_images(args.input)
    
    # output 디렉토리가 없으면 생성
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    # 이미지별 탐지 수행
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue

        # 탐지 수행 및 결과 생성
        annotated_image = detect_and_annotate(yolo_model, image)

        # 결과 출력 또는 저장
        if args.output:
            # output이 디렉터리인 경우 이미지 이름에 확장자를 추가하여 저장
            if os.path.isdir(args.output):
                output_path = os.path.join(args.output, os.path.splitext(os.path.basename(img_path))[0] + ".jpg")
            else:
                output_path = args.output
            
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image to {output_path}")
        else:
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)

    if not args.output:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Object Detection with Optional Optical Flow and Mask Filtering")
    parser.add_argument("--yolo_model", type=str, default="AISolution-main/yolov8x_baram.pt", help="Path to the YOLO model file")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file path")
    args = parser.parse_args()
    
    # CUDA 또는 CPU 디바이스 설정
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)

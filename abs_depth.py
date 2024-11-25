"""
YOLOv8 모델을 이용한 절대 깊이 추정
영상의 사람을 중심으로 깊이 추정
영상의 사람은 2m, 3m, 4m, 5m 깊이로 고정
2m, 3m, 4m, 5m 깊이에 대한 영상 추출하여 DepthAnythingV2 모델로 깊이 추정하여 절대 깊이 보정

"""
import os

import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def calculate_depth_mapping(depth_model, processor, yolo_model, device, calibration_images, order, known_depths=[2, 3, 4, 5]):
    """
    보정 이미지들로부터 상대 깊이와 실제 깊이 간의 매핑 계산
    """
    relative_depths = []
    
    for img in calibration_images:
        # DepthAnything으로 상대 깊이 추정
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            depth_outputs = depth_model(**inputs)
            predicted_depth = depth_outputs.predicted_depth
            depth_map = predicted_depth.squeeze().cpu().numpy()
            
        h, w = img.shape[:2]
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        # 사람 검출
        results = yolo_model(img)[0]
        for detection in results.boxes.data:
            if detection[5] == 0:  # person class (coco)
                x1, y1, x2, y2 = map(int, detection[:4])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                # 사람 영역의 평균 깊이 계산
                person_depth = depth_map[center_y, center_x].mean().item()
                relative_depths.append(person_depth)
                break
    
    # 상대 깊이와 실제 깊이 간의 관계 계산
    relative_depths = np.array(relative_depths)
    known_depths = np.array(known_depths)
    
    # 선형 회귀로 매핑 계수 계산
    if order == 1:
        slope, intercept = np.polyfit(relative_depths, known_depths, order)
        return slope, intercept
    elif order == 2:
        coeffs = np.polyfit(relative_depths, known_depths, 2)  # 2차 다항식 피팅
        return coeffs
    elif order == 3:
        coeffs = np.polyfit(relative_depths, known_depths, 3)  # 3차 다항식 피팅
        return coeffs

def get_center_depth(depth_map, bbox):
    """
    바운딩 박스 중심점의 깊이값 반환
    """
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return depth_map[center_y, center_x].item()

def main(args):
    order = args.order
    device = args.device
    # --- YOLO 모델 로드 ---
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(args.device)
    print(f'Loaded YOLOv8 Model from {args.yolo_model} on {args.device}')
    
    # --- DepthAnythingV2 모델 설정 ---
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model.to(device)
    print('Loading DepthAnythingV2 Model')
    
    # 보정 이미지 로드
    person_dir = "person"
    calibration_images = [
        cv2.imread(os.path.join(person_dir, 'person_2m.jpg')),
        cv2.imread(os.path.join(person_dir, 'person_3m.jpg')),
        cv2.imread(os.path.join(person_dir, 'person_4m.jpg')),
        cv2.imread(os.path.join(person_dir, 'person_5m.jpg'))
    ]
    
    # 깊이 매핑 계수 계산
    slope, intercept, coeffs = None, None, None
    if order == 1:
        slope, intercept = calculate_depth_mapping(depth_model=depth_model,
                                                   processor=processor,
                                                   yolo_model=yolo_model,
                                                   device=device,
                                                   calibration_images=calibration_images,
                                                   order=order)
        print(f"Depth Mapping: depth_meters = {slope:.4f} * relative_depth + {intercept:.4f}")
    else:
        coeffs = calculate_depth_mapping(depth_model=depth_model,
                                         processor=processor,
                                         yolo_model=yolo_model,
                                         device=device,
                                         calibration_images=calibration_images,
                                         order=order)
        print(f"Depth Mapping: depth_meters = {coeffs[0]:.4f} * relative_depth^2 + {coeffs[1]:.4f} * relative_depth + {coeffs[2]:.4f}")
    
    # 실시간 추정에 사용
    video_path = args.input_path
    cap = cv2.VideoCapture(video_path, apiPreference=None)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # 비디오의 기본 정보 가져오기
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 형식
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 동영상 설정
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # DepthAnything으로 상대 깊이 추정
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            depth_outputs = depth_model(**inputs)
            predicted_depth = depth_outputs.predicted_depth
            depth_map = predicted_depth.squeeze().cpu().numpy()
        
        h, w = frame.shape[:2]
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if slope is not None and intercept is not None:
            # 상대 깊이를 절대 깊이로 변환
            depth_map = slope * depth_map + intercept
        elif coeffs is not None:
            if order == 2:
                depth_map = coeffs[0] * depth_map**2 + coeffs[1] * depth_map + coeffs[2]
            elif order == 3:
                depth_map = coeffs[0] * depth_map**3 + coeffs[1] * depth_map**2 + coeffs[2] * depth_map + coeffs[3]
        
        # YOLO로 사람 검출
        results = yolo_model(frame)[0]
        for detection in results.boxes.data:
            if detection[5] == 0:  # person class
                x1, y1, x2, y2 = map(int, detection[:4])
                center_depth = get_center_depth(depth_map, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{center_depth:.2f}m", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if args.output_path:
            out.write(frame)
            cv2.imshow('Absolute Depth Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow('Absolute Depth Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run the models on ('cuda' or 'cpu')",)
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="Path to the YOLOv8 model file")
    parser.add_argument("--order", type=int, default=3, help="Order of the polynomial for depth mapping")
    parser.add_argument("--input_path", type=str, default="person/2m_3m_4m_5m.mp4", help="Path to the input video or folder of images")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save the output video")
    args = parser.parse_args()
    main(args)
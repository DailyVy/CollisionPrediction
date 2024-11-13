import os
import sys
import argparse
from glob import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from tqdm import tqdm
import logging

from utils.ovseg import CATSegSegmentationMap, setup_cfg
from utils.opticalflow import load_unimatch_model, compute_optical_flow, filter_masks_by_avg_flow
from utils.visualize import mask_visualize_and_save, show_anns

from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 설정 로거
def setup_logger():
    """기본 로거 설정."""
    logger = logging.getLogger("OpticalFlowFiltering")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# 마스크 필터링 함수
def mask_overlaps_bbox_x(mask, x_min, x_max):
    """
    마스크의 x축 범위가 BBox의 x축 범위와 겹치는지 확인합니다.

    Args:
        mask (np.ndarray): 2D 이진 마스크 배열.
        x_min (int): BBox의 최소 x좌표.
        x_max (int): BBox의 최대 x좌표.

    Returns:
        bool: 마스크가 BBox의 x축 범위와 겹치면 True, 아니면 False.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False  # 마스크가 비어있는 경우

    mask_x_min = xs.min()
    mask_x_max = xs.max()

    # 마스크의 x축 범위가 BBox의 x축 범위와 겹치는지 확인
    return not (mask_x_max < x_min or mask_x_min > x_max)

# YOLO를 사용한 객체 감지 함수
def detect_bounding_boxes(yolo_model, image, target_classes=None, conf_threshold=0.2, iou_threshold=0.5):
    """
    YOLO 모델을 사용하여 이미지에서 객체를 감지하고 Bounding Box를 반환합니다.

    Args:
        yolo_model (YOLO): 로드된 YOLO 모델.
        image (np.ndarray): BGR 이미지.
        target_classes (List[int], optional): 감지할 클래스의 인덱스 리스트. 기본값은 None (모든 클래스).
        conf_threshold (float): 신뢰도 임계값.
        iou_threshold (float): IoU 임계값.

    Returns:
        List[dict]: Bounding Box와 클래스 정보를 포함한 리스트. 예: [{'bbox': (x_min, y_min, x_max, y_max), 'class': 'Hoist_hook'}, ...]
    """
    results = yolo_model(image, conf=conf_threshold, iou=iou_threshold, device=yolo_model.device)[0]
    bboxes = []
    for detection in results.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cls_idx = int(detection.cls)
        conf = float(detection.conf)
        label = results.names[cls_idx]
        if target_classes is None or cls_idx in target_classes:
            bboxes.append({
                'bbox': (x1, y1, x2, y2),
                'class': label,
                'confidence': conf
            })
    return bboxes

# 메인 함수
def main(args):
    # --- 입력 경로 검증 및 처리 ---
    if not args.input:
        print("No input provided. Please specify an input path using --input.")
        exit(1)
    
    input_paths = []
    input_path = args.input[0]
    print(f"Processing input: {input_path}")

    if os.path.isdir(input_path):
        # 디렉토리인 경우, 디렉토리 내의 모든 이미지 파일을 리스트로 가져옴
        input_paths = sorted(
            glob(os.path.join(os.path.expanduser(input_path), "*.png")) + 
            glob(os.path.join(os.path.expanduser(input_path), "*.jpg")) +
            glob(os.path.join(os.path.expanduser(input_path), "*.jpeg"))
        )
        assert input_paths, f"No image files found in directory: {input_path}"
    elif os.path.isfile(input_path):
        # 단일 파일인 경우, 해당 파일을 리스트로 처리
        input_paths = args.input
    else:
        raise ValueError(f"Input path is neither a directory nor a file: {input_path}")

    # --- 로깅 설정 ---
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # --- YOLO 모델 로드 ---
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(args.device)
    print(f'Loaded YOLOv8 Model from {args.yolo_model} on {args.device}')

    # --- SAM 및 UniMatch 모델 로드 ---
    device = args.device
    
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    print(f'Loaded Segment Anything Model: vit_h')
    
    flow_model = load_unimatch_model(args.flow_checkpoint, device=device)
    
    # --- Optical Flow 인자 설정 ---
    flow_args = argparse.Namespace(
        padding_factor=32,         # 패딩 팩터 설정
        inference_size=None,       # 추론 크기
        attn_type='swin',
        attn_splits_list=[2, 8],   # attention split list
        corr_radius_list=[-1, 4],  # correlation 반경 리스트
        prop_radius_list=[-1, 1],  # propagation 반경 리스트
        num_reg_refine=6,           # refinement 스텝 개수
        task='flow'
    )

    # --- CAT-Seg 설정 ---
    cfg = setup_cfg(args)
    catseg_map = CATSegSegmentationMap(cfg)
    
    text = 'floor, person, forklift, machine, wall, ceiling' 
    target_class = 0  # floor's index: 0
    
    # --- 출력 디렉토리 설정 ---
    if args.output:
        pre_flow_dir = os.path.join(args.output, "pre_flow")
        post_flow_dir = os.path.join(args.output, "post_flow")
        os.makedirs(pre_flow_dir, exist_ok=True)
        os.makedirs(post_flow_dir, exist_ok=True)
    else:
        pre_flow_dir = None
        post_flow_dir = None

    for idx, path in enumerate(tqdm(input_paths, desc="Processing Images")):
        # try:
            # 이미지 로드
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {path}")
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 다음 이미지 로드 (Optical Flow를 위해)
            if idx < len(input_paths) - 1:
                next_path = input_paths[idx + 1]
                next_img = cv2.imread(next_path)
                if next_img is None:
                    raise FileNotFoundError(f"Next image not found at path: {next_path}")
                next_image = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)
            else:
                next_image = None

            start_time = time.time()

            # CAT-Seg를 사용한 세그멘테이션
            predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)

            # SAM을 사용한 마스크 생성
            mask_generator_custom = SamAutomaticMaskGeneratorCustom(sam, semantic_map=segmap, target_class=target_class)
            masks = mask_generator_custom.generate(image)
            print(f"Number of masks: {len(masks)}")
            if len(masks) == 0:
                print("No masks generated. Skipping this image.")
                continue

            # YOLO를 사용한 Bounding Box 감지
            # 모든 클래스가 관심 객체이므로 target_classes는 None으로 설정 (모든 클래스)
            bboxes = detect_bounding_boxes(
                yolo_model, 
                img, 
                target_classes=None, 
                conf_threshold=args.yolo_conf, 
                iou_threshold=args.yolo_iou
            )
            print(f"Detected {len(bboxes)} bounding boxes")
            
            if len(bboxes) == 0: # todos. 오히려 마스크를 모두 삭제해야함
                print("No bounding boxes detected. 모든 마스크를 유지합니다.")
                filtered_masks = masks
            else:
                # Hoist_hook 클래스에 해당하는 Bounding Box만 사용하여 마스크 필터링
                filtered_masks = []
                hoist_bboxes = [bbox_info for bbox_info in bboxes if bbox_info['class'] == 'Hoist']
                
                if not hoist_bboxes: # todos. 오히려 마스크를 모두 삭제해야함
                    print("No Hoist_hook bounding boxes detected. 모든 마스크를 유지합니다.")
                    filtered_masks = masks
                else:
                    for bbox_info in hoist_bboxes:
                        bbox = bbox_info['bbox']
                        x_min, y_min, x_max, y_max = bbox
                        masks_in_bbox = [mask for mask in masks if mask_overlaps_bbox_x(mask['segmentation'], x_min, x_max)]
                        filtered_masks.extend(masks_in_bbox)
                    # 중복 제거
                    filtered_masks = list({id(mask): mask for mask in filtered_masks}.values())
                    print(f"Total masks after Hoist_hook BBox filtering: {len(filtered_masks)}")

            # Optical Flow 계산
            if next_image is not None:
                image1_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                image2_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float()
                flow_pr, flow_magnitude_resized = compute_optical_flow(
                    flow_model, 
                    image1_tensor, 
                    image2_tensor,
                    flow_args,
                    device
                )
                
                print(f"type(flow_pr): {type(flow_pr)}, shape: {flow_pr.shape}")
                print(f"flow_magnitude_resized shape: {flow_magnitude_resized.shape}")
            else:
                print("다음 이미지가 없어 Optical Flow를 계산할 수 없습니다.")
                flow_magnitude_resized = None

            # Optical Flow 기반 마스크 필터링
            if flow_magnitude_resized is not None:
                threshold_flow_filter = args.threshold_flow
                final_filtered_masks = filter_masks_by_avg_flow(filtered_masks, flow_magnitude_resized, threshold=threshold_flow_filter)
            else:
                final_filtered_masks = []
                print("Optical Flow가 존재하지 않아 모든 마스크를 제거합니다.")
                
            # --- 각 마스크의 평균 Flow Magnitude 계산 ---
            mask_flow_magnitudes = []
            for mask in final_filtered_masks:
                # 마스크 영역의 Flow Magnitude 추출
                flow_in_mask = flow_magnitude_resized[mask['segmentation'] > 0]
                if flow_in_mask.size > 0:
                    avg_flow = np.mean(flow_in_mask)
                else:
                    avg_flow = 0.0
                mask_flow_magnitudes.append(avg_flow)
            print(f"mask_flow_magnitudes : {mask_flow_magnitudes}")
            
            # --- 시각화 및 저장 ---
            if args.output:
                # 파일 이름 설정
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                
                # Optical Flow 적용 전의 마스크 시각화 및 저장
                if pre_flow_dir:
                    pre_flow_save_path = os.path.join(pre_flow_dir, f"{name}_filtered_masks{ext}")
                    # filtered_masks에 'avg_flow' 추가
                    for idx, mask in enumerate(filtered_masks):
                        mask['avg_flow'] = mask_flow_magnitudes[idx] if idx < len(mask_flow_magnitudes) else 0.0
                        mask['class'] = 'Unknown'  # 클래스 정보가 없으므로 필요시 추가
                    mask_visualize_and_save(
                        image=image,
                        masks=filtered_masks,
                        save_path=pre_flow_save_path,
                        title="Filtered Masks Before Flow Filtering"
                    )
                
                # Optical Flow 적용 후의 마스크 시각화 및 저장
                if post_flow_dir:
                    post_flow_save_path = os.path.join(post_flow_dir, f"{name}_final_filtered_masks{ext}")
                    # final_filtered_masks에 'avg_flow' 추가
                    final_flow_magnitudes = [mask_flow_magnitudes[idx] for idx in range(len(final_filtered_masks))]
                    mask_visualize_and_save(
                        image=image,
                        masks=final_filtered_masks,
                        save_path=post_flow_save_path,
                        title="Final Filtered Masks After Flow Filtering"
                    )
            else:
                # 시각화만 표시
                # Optical Flow 적용 전의 마스크 시각화
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_anns(filtered_masks, alpha=0.5)
                
                plt.axis('off')
                plt.title("Filtered Masks Before Flow Filtering")
                plt.show()
                plt.close()
                
                # Optical Flow 적용 후의 마스크 시각화
                if final_filtered_masks:
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    show_anns(final_filtered_masks, alpha=0.5)
                    
                    plt.axis('off')
                    plt.title("Final Filtered Masks After Flow Filtering")
                    plt.show()
                    plt.close()       
                
            # --- 로깅 ---
            logger.info(
                "{}: Processed in {:.2f}s".format(
                    path,
                    time.time() - start_time
                )
            )
            print(f"\nProcessed {path}\n")

        # except Exception as e:
        #     print(f"Error processing {path}: {e}")
        #     logger.error(f"Error processing {path}: {e}")
        #     continue  # 다음 이미지로 넘어감

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow and Mask Filtering with YOLOv8 Integration")
    parser.add_argument(
        "--config-file",
        default="CAT-Seg/configs/vitl_336_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs='+', required=True, help="Input image path(s) or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file path")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth", type=str, help="SAM model checkpoint path")
    parser.add_argument("--flow_checkpoint", default="unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", type=str, help="UniMatch Optical Flow model checkpoint path")
    parser.add_argument("--threshold_flow", type=float, default=2.0, help="Optical flow threshold value")
    
    # YOLO 관련 인자 추가
    parser.add_argument("--yolo_model", type=str, default="AISolution-main/yolov8x_baram.pt", help="Path to the YOLOv8 model file")
    parser.add_argument("--yolo_conf", type=float, default=0.2, help="YOLO 신뢰도 임계값")
    parser.add_argument("--yolo_iou", type=float, default=0.5, help="YOLO IoU 임계값")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run the models on ('cuda' or 'cpu')",)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)

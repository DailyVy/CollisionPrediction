import os
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
import json
from datetime import datetime

from utils.distance_utils import create_distance_database, calculate_pixel_scale_from_person, print_distance_database
from utils.ovseg import CATSegSegmentationMap, setup_cfg
from utils.opticalflow import load_unimatch_model, compute_optical_flow, filter_masks_by_avg_flow
from utils.depth import get_depth_at_position, get_3d_positions, process_depth
from utils.visualize import visualize_with_3d_positions, visualize_with_distances
from utils.mask_utils import mask_overlaps_bbox_x, get_mask_position, merge_overlapping_masks
from utils.detection_utils import detect_bounding_boxes, get_bbox_center

from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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

def test_pixel_scale(objects, depth_map, image_height=360):
    """
    첫 번째 감지된 사람을 기준으로 픽셀 스케일 계산
    """
    person_objects = [obj for obj in objects if obj['class'] == 'Person']
    if person_objects:
        first_person = person_objects[0]
        scale_info = calculate_pixel_scale_from_person(first_person['bbox'], image_height)
        
        print("\n" + "="*50)
        print("📏 픽셀 스케일 테스트 결과")
        print("="*50)
        print(f"위치: 화면 아래에서 {scale_info['relative_position']*100:.1f}% 지점")
        print(f"보정 계수: {scale_info['correction_factor']:.2f}")
        print(f"원본 픽셀 스케일: {scale_info['original_pixels_per_meter']:.2f} pixels/meter")
        print(f"보정된 픽셀 스케일: {scale_info['pixels_per_meter']:.2f} pixels/meter")
        
        # depth 값도 함께 출력하여 비교
        x, y = get_bbox_center(first_person['bbox'])
        depth_value = depth_map[int(y), int(x)]
        print(f"해당 위치의 depth 값: {depth_value}")
        print("="*50 + "\n")
        
        return scale_info
    return None

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
    
    # --- DepthAnythingV2 모델 설정 ---
    print('Loading DepthAnythingV2 Model')
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.to(device)

    # --- 출력 디렉토리 설정 ---
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    for idx, path in enumerate(tqdm(input_paths, desc="이미지 처리 중")):
        stage_times = {}
        total_start_time = time.time()
        
        # 이미지 로드
        stage_start = time.time()
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(f"Image not found at path: {path}")
        current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stage_times['이미지 로드'] = time.time() - stage_start

        # 다음 이미지 로드
        stage_start = time.time()
        if idx < len(input_paths) - 1:
            next_path = input_paths[idx + 1]
            next_img = cv2.imread(next_path)
            if next_img is None: raise FileNotFoundError(f"Next image not found at path: {next_path}")
            next_image = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)
        else:
            next_image = None
            break
        stage_times['다음 이미지 로드'] = time.time() - stage_start

        # CAT-Seg 세그멘테이션
        stage_start = time.time()
        predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)
        stage_times['CAT-Seg 세그멘테이션'] = time.time() - stage_start

        # SAM 마스크 생성
        stage_start = time.time()
        mask_generator_custom = SamAutomaticMaskGeneratorCustom(sam, semantic_map=segmap, target_class=target_class)
        masks = mask_generator_custom.generate(current_image)
        stage_times['SAM 마스크 생성'] = time.time() - stage_start

        # YOLO 객체 감지
        stage_start = time.time()
        objects = detect_bounding_boxes(yolo_model, img, target_classes=None, 
                                      conf_threshold=args.yolo_conf, iou_threshold=args.yolo_iou)
        stage_times['YOLO 객체 감지'] = time.time() - stage_start

        # Hoist 객체 확인
        hoist_objects = [obj for obj in objects if obj['class'] == 'Hoist']
        if not hoist_objects:
            print("No Hoist detected. 다음 이미지로 넘어갑니다.")
            continue

        # Hoist bbox를 기준으로 마스크 필터링
        filtered_masks = []
        for hoist in hoist_objects:
            x_min, y_min, x_max, y_max = hoist['bbox']
            masks_in_bbox = [mask for mask in masks if mask_overlaps_bbox_x(mask['segmentation'], x_min, x_max)]
            filtered_masks.extend(masks_in_bbox)

        # 중복 제거
        filtered_masks = list({id(mask): mask for mask in filtered_masks}.values())

        # Optical Flow 계산
        stage_start = time.time()
        if next_image is not None:
            image1_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float()
            image2_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float()
            flow_pr, flow_magnitude_resized = compute_optical_flow(flow_model, image1_tensor, image2_tensor, flow_args, device)
        stage_times['Optical Flow 계산'] = time.time() - stage_start

        # Optical flow 계산 및 마스크 필터링
        if next_image is not None:
            if flow_magnitude_resized is not None:
                final_filtered_masks = filter_masks_by_avg_flow(filtered_masks, flow_magnitude_resized, threshold=args.threshold_flow)
                final_filtered_masks = merge_overlapping_masks(final_filtered_masks)
            else:
                print("Optical Flow가 존재하지 않아 모든 마스크 제거합니다.")
                continue
        else:
            print("다음 이미지가 없습니다. Optical Flow를 계산할 수 없습니다.")
            continue

        # Heavy object 위치 계산
        heavy_object_info = None
        if final_filtered_masks:
            merged_mask = final_filtered_masks[0]
            x_pos, y_pos = get_mask_position(merged_mask)
            heavy_object_info = {
                'position': (x_pos, y_pos),
                'mask': merged_mask,
                'bbox': hoist_objects[0]['bbox'],
                'confidence': hoist_objects[0]['confidence']
            }
            print(f"Heavy object detected at position: ({x_pos:.2f}, {y_pos:.2f})")
        else:
            print("No heavy object detected after filtering")
            continue

        # Person과 Forklift 정보 추출
        person_info = [{
            'position': get_bbox_center(obj['bbox']),
            'bbox': obj['bbox'],
            'confidence': obj['confidence']
        } for obj in objects if obj['class'] == 'Person']

        forklift_info = [{
            'position': get_bbox_center(obj['bbox']),
            'bbox': obj['bbox'],
            'confidence': obj['confidence']
        } for obj in objects if obj['class'] == 'Forklift']

        # Depth Estimation
        stage_start = time.time()
        depth_map = process_depth(current_image, model, processor)
        stage_times['Depth Estimation'] = time.time() - stage_start

        # 3D 위치 정보 계산
        stage_start = time.time()
        heavy_object_3d = get_3d_positions([heavy_object_info], depth_map) if heavy_object_info else None
        person_3d = get_3d_positions(person_info, depth_map)
        forklift_3d = get_3d_positions(forklift_info, depth_map)
        stage_times['3D 위치 계산'] = time.time() - stage_start

        # 데이터베이스 생성
        stage_start = time.time()
        distance_db = create_distance_database(
            heavy_object_3d,
            person_3d,
            forklift_3d
        )
        stage_times['데이터베이스 생성'] = time.time() - stage_start
        
        # 현재 시간 추가
        distance_db['frame_info']['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print_distance_database(distance_db)
        
        # 시각화
        if args.output:
            json_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(path))[0]}_distances.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(distance_db, f, indent=4, ensure_ascii=False)
            print(f"\n💾 데이터베이스가 저장되었습니다: {json_path}")
            
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_distances{ext}")
            
            visualize_with_distances(
                image=current_image,
                heavy_objects_positions_3d=heavy_object_3d if heavy_object_3d else [],
                person_positions_3d=person_3d,
                forklift_positions_3d=forklift_3d,
                distances=distance_db['distances'],
                masks=final_filtered_masks,
                title="3D Positions and Distances"
            )
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # JSON 저장
            json_path = os.path.join(args.output, f"{name}_distances.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(distance_db, f, indent=4, ensure_ascii=False)
        else:
            visualize_with_distances(
                image=current_image,
                heavy_objects_positions_3d=heavy_object_3d if heavy_object_3d else [],
                person_positions_3d=person_3d,
                forklift_positions_3d=forklift_3d,
                distances=distance_db['distances'],
                masks=final_filtered_masks,
                title="3D Positions and Distances"
            )
            plt.show()
            plt.close()
            
        total_time = time.time() - total_start_time
        
        # 처리 시간 출력
        print("\n" + "="*50)
        print(f"🕒 처리 시간 분석 - {os.path.basename(path)}")
        print("="*50)
        for stage, duration in stage_times.items():
            print(f"{stage:<25}: {duration:>6.2f}초 ({duration/total_time*100:>5.1f}%)")
        print("-"*50)
        print(f"{'총 처리 시간':<25}: {total_time:>6.2f}초 (100.0%)")
        print("="*50 + "\n")

        # 로깅
        logger.info(f"{path}: 총 처리 시간 {total_time:.2f}초")

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

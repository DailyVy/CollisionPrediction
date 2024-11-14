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

from utils.ovseg import CATSegSegmentationMap, setup_cfg
from utils.opticalflow import load_unimatch_model, compute_optical_flow, filter_masks_by_avg_flow
from utils.depth import get_depth_at_position, get_3d_positions
from utils.visualize import mask_visualize_and_save, show_anns, visualize_with_3d_positions, visualize_with_positions
from utils.mask_utils import mask_overlaps_bbox_x, get_mask_position, merge_overlapping_masks
from utils.detection_utils import detect_bounding_boxes, get_bbox_center

from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from transformers import pipeline # DepthAnythingV2


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
    
    # # --- DepthAnythingV2 인자 설정 ---
    print('Loading DepthAnythingV2 Model')
    depth_model = pipeline(task='depth-estimation',
                           model='depth-anything/Depth-Anything-V2-Small-hf', device=0)
    
    # --- 출력 디렉토리 설정 ---
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    for idx, path in enumerate(tqdm(input_paths, desc="Processing Images")):
        # 현재 이미지 로드
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 다음 이미지 로드 (Optical Flow를 위해)
        if idx < len(input_paths) - 1:
            next_path = input_paths[idx + 1]
            next_img = cv2.imread(next_path)
            if next_img is None:
                raise FileNotFoundError(f"Next image not found at path: {next_path}")
            next_image = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)
        else:
            next_image = None
            break
        
        start_time = time.time()

        # CAT-Seg를 사용한 세그멘테이션
        predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)

        # SAM을 사용한 마스크 생성
        mask_generator_custom = SamAutomaticMaskGeneratorCustom(sam, semantic_map=segmap, target_class=target_class)
        masks = mask_generator_custom.generate(current_image)
        print(f"Number of masks: {len(masks)}")
        if len(masks) == 0:
            print("No masks generated. Skipping this image.")
            continue
                      
        # YOLO를 사용한 Bounding Box 감지
        # 모든 클래스가 관심 객체이므로 target_classes는 None으로 설정 (모든 클래스)
        objects = detect_bounding_boxes(
            yolo_model, 
            img, 
            target_classes=None, 
            conf_threshold=args.yolo_conf, 
            iou_threshold=args.yolo_iou
        )
        
        class_bboxes = {'Hoist': [],
                        'Person': [],
                        'Magnet': [],
                        'Forklift': []}
        
        for object_info in objects:
            if object_info['class'] in class_bboxes:
                class_bboxes[object_info['class']].append(object_info)
                        
        hoist_bboxes = class_bboxes['Hoist']
        
        if not hoist_bboxes:
            print("No Hoist bounding boxes detected. 모든 마스크를 삭제하고 다음 이미지로 넘어갑니다.")
            continue

        # Hoist 클래스에 해당하는 Bounding Box를 사용하여 마스크 필터링
        filtered_masks = []
        for bbox_info in hoist_bboxes:
            bbox = bbox_info['bbox']
            x_min, y_min, x_max, y_max = bbox
            masks_in_bbox = [mask for mask in masks if mask_overlaps_bbox_x(mask['segmentation'], x_min, x_max)]
            filtered_masks.extend(masks_in_bbox)

        # 중복 제거
        filtered_masks = list({id(mask): mask for mask in filtered_masks}.values())

        # 다음 이미지가 있는지 확인하고 optical flow 계산
        # 이미 로드된 next_image가 있으면 optical flow 계산
        if next_image is not None:
            image1_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float()
            image2_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float()
            flow_pr, flow_magnitude_resized = compute_optical_flow(flow_model, image1_tensor, image2_tensor, flow_args, device)
            
            print(f"type(flow_pr): {type(flow_pr)}, shape: {flow_pr.shape}")
            print(f"flow_magnitude_resized shape: {flow_magnitude_resized.shape}")
        else:
            print("다음 이미지가 없습니다. Optical Flow를 계산할 수 없습니다.")
            flow_magnitude_resized = None
            break

        # Optical Flow 기반 마스크 필터링
        if flow_magnitude_resized is not None:
            final_filtered_masks = filter_masks_by_avg_flow(filtered_masks, flow_magnitude_resized, threshold=args.threshold_flow)
            # 마스크 병합 추가
            final_filtered_masks = merge_overlapping_masks(final_filtered_masks)
        else:
            filtered_masks = []
            print("Optical Flow가 존재하지 않아 모든 마스크를 제거합니다.")
            continue
        
        # final_filtered_masks에서 각 heavy object의 위치 계산
        heavy_objects_positions = []
        for mask in final_filtered_masks:
            x_pos, y_pos = get_mask_position(mask)
            heavy_objects_positions.append({
                'mask_id': id(mask),  # 마스크 식별을 위한 ID
                'position': (x_pos, y_pos),
                'mask': mask  # 원본 마스크 정보 보존
            })
                
        print(f"Detected {len(heavy_objects_positions)} heavy objects:")
        for obj in heavy_objects_positions:
            print(f"Heavy object at position: ({obj['position'][0]:.2f}, {obj['position'][1]:.2f})")
                        
        # Person과 Forklift의 위치 계산
        person_positions = []
        for person in class_bboxes['Person']:
            center_x, center_y = get_bbox_center(person['bbox'])
            person_positions.append({
                'position': (center_x, center_y),
                'bbox': person['bbox']  # 원본 bbox 정보 보존
            })

        forklift_positions = []
        for forklift in class_bboxes['Forklift']:
            center_x, center_y = get_bbox_center(forklift['bbox'])
            forklift_positions.append({
                'position': (center_x, center_y),
                'bbox': forklift['bbox']  # 원본 bbox 정보 보존
            })
                
        if person_positions:
            print(f"\nDetected {len(person_positions)} people:")
            for idx, person in enumerate(person_positions):
                print(f"Person {idx + 1} at position: ({person['position'][0]:.2f}, {person['position'][1]:.2f})")

        if forklift_positions:
            print(f"\nDetected {len(forklift_positions)} forklifts:")
            for idx, forklift in enumerate(forklift_positions):
                print(f"Forklift {idx + 1} at position: ({forklift['position'][0]:.2f}, {forklift['position'][1]:.2f})")
                
        # Depth Estimation 수행
        pil_image = Image.fromarray(current_image)
        depth_output = depth_model(pil_image)
        depth_map = np.array(depth_output['depth'])

        # 3D 위치 정보 계산
        heavy_objects_positions_3d = get_3d_positions(heavy_objects_positions, depth_map)
        person_positions_3d = get_3d_positions(person_positions, depth_map)
        forklift_positions_3d = get_3d_positions(forklift_positions, depth_map)

        # 결과 출력
        print("\nObject positions with depth information:")
        print("\nHeavy Objects:")
        for idx, obj in enumerate(heavy_objects_positions_3d):
            x, y, z = obj['position_3d']
            print(f"Heavy object {idx + 1} position: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f}m)")

        print("\nPeople:")
        for idx, person in enumerate(person_positions_3d):
            x, y, z = person['position_3d']
            print(f"Person {idx + 1} position: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f}m)")

        print("\nForklifts:")
        for idx, forklift in enumerate(forklift_positions_3d):
            x, y, z = forklift['position_3d']
            print(f"Forklift {idx + 1} position: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f}m)")
                    
        # 시각화 코드 수정
        if args.output:
            # 파일 이름 설정
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            
            # Optical Flow 적용 후의 마스크와 위치 시각화 및 저장
            post_flow_save_path = os.path.join(args.output, f"{name}_final_filtered_masks_3d{ext}")
            fig = visualize_with_3d_positions(
                image=current_image,
                final_filtered_masks=final_filtered_masks,
                heavy_objects_positions_3d=heavy_objects_positions_3d,
                person_positions_3d=person_positions_3d,
                forklift_positions_3d=forklift_positions_3d,
                title="Object Positions with Depth Information"
            )
            fig.savefig(post_flow_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            visualize_with_positions(
                image=current_image,
                final_filtered_masks=final_filtered_masks,
                person_positions=person_positions,
                forklift_positions=forklift_positions,
                title="Final Filtered Masks and Object Positions (After Flow)"
            )
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

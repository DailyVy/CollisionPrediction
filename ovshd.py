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

from transformers import pipeline # DepthAnythingV2
from utils.ovseg import CATSegSegmentationMap, setup_cfg
from utils.visualize import depth_visualize_and_save, show_anns
from utils.depth import filter_masks_by_depth

from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom, SamPredictor

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

# Bounding Box 그리기 함수
def draw_bounding_box(image):
    """
    사용자로부터 Bounding Box를 그리게 하는 함수.

    Args:
        image (np.ndarray): RGB 이미지.

    Returns:
        bbox (tuple): Bounding Box 좌표 (x_min, y_min, x_max, y_max).
    """
    drawing = False  # 마우스 버튼이 눌렸는지 여부
    start_point = (-1, -1)  # BBox의 시작점
    end_point = (-1, -1)  # BBox의 끝점
    image_copy = image.copy()

    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_point, end_point, drawing, image_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                image_copy = image.copy()
                cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

    cv2.namedWindow('Draw BBox')
    cv2.setMouseCallback('Draw BBox', draw_rectangle)
    
    print("이미지에 Bounding Box를 그려주세요. 완료되면 's' 키를 눌러주세요.")

    while True:
        # OpenCV 창에 이미지 표시
        cv2.imshow('Draw BBox', cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # 's' 키를 누르면 그리기 완료
            break
        elif key == 27:  # 'Esc' 키를 누르면 종료
            cv2.destroyAllWindows()
            exit()
    
    cv2.destroyAllWindows()
    
    # BBox 좌표 추출
    if start_point != (-1, -1) and end_point != (-1, -1):
        x1, y1 = start_point
        x2, y2 = end_point
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        print(f"Selected BBox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        return (x_min, y_min, x_max, y_max)
    else:
        print("BBox가 선택되지 않았습니다. 모든 마스크를 유지합니다.")
        return None

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

    # --- SAM 및 UniMatch 모델 로드 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    print(f'Loaded Segment Anything Model: vit_h')
    
    predictor = SamPredictor(sam) 

    # --- DepthAnythingV2 인자 설정 ---
    print('Loading DepthAnythingV2 Model')
    depth_model = pipeline(task='depth-estimation',
                           model='depth-anything/Depth-Anything-V2-Small-hf', device=0)

    # --- CAT-Seg 설정 ---
    cfg = setup_cfg(args)
    catseg_map = CATSegSegmentationMap(cfg)
    
    text = 'floor, person, forklift, machine, wall, ceiling' 
    target_class = 0  # floor's index: 0
    
    # --- 출력 디렉토리 설정 ---
    if args.output:
        depth_dir = os.path.join(args.output, "depth_filtered_masks")
        os.makedirs(depth_dir, exist_ok=True)
    else:
        depth_dir = None

    for idx, path in enumerate(tqdm(input_paths, desc="Processing Images")):
        # try:
            # 이미지 로드
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {path}")
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image) # for DepthAnythingV2

            start_time = time.time()
            
            # Depth Estimation 수행
            depth_array = depth_model(pil_image) 
            depth_mag = np.array(depth_array['depth'])  # DepthAnythingV2의 출력 형식에 따라 조정 필요

            # CAT-Seg를 사용한 세그멘테이션
            predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)

            # SAM을 사용한 마스크 생성
            mask_generator_custom = SamAutomaticMaskGeneratorCustom(sam, semantic_map=segmap, target_class=target_class)
            masks = mask_generator_custom.generate(image)
            print(f"Number of masks: {len(masks)}")
            if len(masks) == 0:
                print("No masks generated. Skipping this image.")
                continue

            bbox = draw_bounding_box(image)

            # Bounding Box를 기준으로 마스크 필터링
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                filtered_masks = [mask for mask in masks if mask_overlaps_bbox_x(mask['segmentation'], x_min, x_max)]
            else:
                filtered_masks = masks
            print(f"Total masks: {len(masks)}, Masks after BBox filtering: {len(filtered_masks)}")
            
            # Bounding Box 내에서 호이스트 마스크 생성 및 Depth 평균 계산
            if bbox is not None:
                # SAM을 사용하여 Bounding Box 내에서 호이스트 마스크 생성
                predictor.set_image(image)
                boxes = np.array([bbox])
                transformed_box = predictor.transform.apply_boxes_torch(
                    torch.as_tensor(boxes, dtype=torch.float, device=predictor.device),
                    image.shape[:2])
                masks_pred, scores_pred, logits_pred = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_box,
                    multimask_output=False
                )
                hoist_mask = masks_pred.cpu().numpy()  
                hoist_mask_squeezed = np.squeeze(hoist_mask)
                if hoist_mask_squeezed.ndim > 2:
                    hoist_mask_squeezed = hoist_mask_squeezed[..., 0]

                # 호이스트 마스크의 Depth 평균 계산
                depth_in_hoist = depth_mag[hoist_mask_squeezed]
                if depth_in_hoist.size > 0:
                    bbox_depth_mean = np.mean(depth_in_hoist)
                else:
                    bbox_depth_mean = 0.0
                print(f"Hoist Mask Depth Mean: {bbox_depth_mean}")
            else:
                bbox_depth_mean = None
                print("Bounding Box Depth Mean: None")
            
            # # Bounding Box의 평균 Depth 계산
            # if bbox is not None:
            #     bbox_depth_region = depth_mag[y_min:y_max, x_min:x_max]
            #     if bbox_depth_region.size > 0:
            #         bbox_depth_mean = np.mean(bbox_depth_region)
            #     else:
            #         bbox_depth_mean = 0.0
            # else:
            #     bbox_depth_mean = None
            # print(f"Bounding Box Depth Mean: {bbox_depth_mean}")
            
            # Depth 기반 마스크 필터링
            if bbox_depth_mean is not None:
                threshold_depth = 0.1  # Depth 유사성 임계값 (10%)
                final_filtered_masks = filter_masks_by_depth(filtered_masks, depth_mag, bbox_depth_mean, threshold=threshold_depth)
            else:
                final_filtered_masks = []
                print("Bounding Box Depth가 없어서 마스크 필터링을 수행하지 않습니다.")
                
            # --- 각 마스크의 평균 Depth 계산 ---
            mask_depth_means = []
            for mask in filtered_masks:
                depth_in_mask = depth_mag[mask['segmentation'] > 0]
                if depth_in_mask.size > 0:
                    avg_depth = np.mean(depth_in_mask)
                else:
                    avg_depth = 0.0
                mask_depth_means.append(avg_depth)
            print(f"mask_depth_means : {mask_depth_means}")

            # --- Depth 기반 마스크 필터링 결과 저장 ---
            if args.output:
                # 파일 이름 설정
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                
                depth_save_path = os.path.join(depth_dir, f"{name}_depth_filtered_masks{ext}")
                
                # option for hoist visualization
                hoist = dict()
                hoist["segmentation"] = hoist_mask_squeezed
                hoist["area"] = 100
                final_filtered_masks.append(hoist)
                
                # final_filtered_masks에 'avg_depth' 추가 (이미 추가됨 in filter_masks_by_depth)
                depth_visualize_and_save(
                    image=image,
                    final_masks=final_filtered_masks,
                    masks=filtered_masks,
                    mask_depth_magnitudes=mask_depth_means,  
                    bbox=bbox,
                    avg_depth_bbox=bbox_depth_mean,  # 시각화 함수는 'avg_flow_bbox'를 사용하지만 Depth로 대체
                    save_path=depth_save_path,
                    title="Depth Filtered Masks"
                )
            else:
                # Depth 필터링된 마스크 시각화
                plt.figure(figsize=(10,10))
                image_with_bboxes = image.copy()
                if bbox is not None:
                    cv2.rectangle(image_with_bboxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                plt.imshow(image_with_bboxes)
                show_anns(final_filtered_masks, alpha=0.5)
                
                ax = plt.gca()

                # 각 마스크에 평균 Depth 표시
                for idx, mask in enumerate(final_filtered_masks):
                    avg_depth = mask_depth_means[idx]
                    mask_mask = mask['segmentation']
                    ys, xs = np.nonzero(mask_mask)
                    if len(xs) > 0 and len(ys) > 0:
                        min_x = xs.min()
                        min_y = ys.min()
                        # 텍스트 표시 (왼쪽 상단에 약간의 오프셋 추가)
                        ax.text(min_x + 5, min_y + 5, f"{avg_depth:.2f}", color='white', fontsize=8, weight='bold',
                                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

                # BBox의 평균 Depth 표시
                if bbox_depth_mean is not None:
                    # BBox의 상단 좌표 계산
                    bbox_min_x = bbox[0]
                    bbox_min_y = bbox[1]
                    # 텍스트 표시
                    ax.text(bbox_min_x + 5, bbox_min_y + 5, f"BBox Avg Depth: {bbox_depth_mean:.2f}", color='yellow', fontsize=12, weight='bold',
                            ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

                plt.axis('off')
                plt.title("Depth Filtered Masks")
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
    parser = argparse.ArgumentParser(description="Optical Flow and Mask Filtering")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs='+', required=True, help="Input image path(s) or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file path")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth", type=str, help="SAM model checkpoint path")
    parser.add_argument(
        "--threshold_depth",
        type=float,
        default=0.1,
        help="Threshold for matching with hoist's depth",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)

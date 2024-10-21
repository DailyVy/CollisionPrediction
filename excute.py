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

from utils.OVSeg import CATSegSegmentationMap, setup_cfg
from utils.OpticalFlow import load_unimatch_model, compute_optical_flow, calculate_flow_magnitude, resize_flow_magnitude, visualize_flow
from utils.visualize import visualize_and_save, show_anns

from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from unimatch.utils.flow_viz import flow_to_image

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

# Optical Flow 기반 마스크 필터링 함수
def mask_has_flow(mask, flow_mag, threshold=1.0):
    """
    마스크 영역 내의 평균 Optical Flow가 임계값을 초과하는지 확인합니다.

    Args:
        mask (np.ndarray): 2D 이진 마스크 배열.
        flow_mag (np.ndarray): Optical Flow Magnitude 배열.
        threshold (float): Flow magnitude 평균 임계값.

    Returns:
        bool: 마스크 영역 내의 평균 flow magnitude가 임계값을 초과하면 True, 아니면 False.
    """
    # 마스크 영역 내의 Optical Flow Magnitude 추출
    flow_in_mask = flow_mag[mask > 0]
    if flow_in_mask.size == 0:
        return False
    # 평균 Flow Magnitude 계산
    avg_flow = np.mean(flow_in_mask)
    # 평균이 임계값을 초과하는지 확인
    return avg_flow > threshold

def filter_masks_by_avg_flow(masks, flow_magnitude, threshold=1.0):
    """
    마스크의 평균 Optical Flow를 기반으로 마스크를 필터링합니다.

    Args:
        masks (list): 마스크 리스트.
        flow_magnitude (np.ndarray): Optical Flow Magnitude 배열.
        threshold (float): Flow magnitude 평균 임계값.

    Returns:
        list: 평균 Flow Magnitude가 임계값을 초과하는 마스크 리스트.
    """
    final_filtered_masks = []
    for mask in masks:
        if mask_has_flow(mask['segmentation'], flow_magnitude, threshold):
            final_filtered_masks.append(mask)
    print(f"Optical Flow 평균을 고려한 최종 마스크 수: {len(final_filtered_masks)} (Threshold: {threshold})")
    return final_filtered_masks


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
        try:
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

            bbox = draw_bounding_box(image)

            # Bounding Box를 기준으로 마스크 필터링
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                filtered_masks = [mask for mask in masks if mask_overlaps_bbox_x(mask['segmentation'], x_min, x_max)]
            else:
                filtered_masks = masks
            print(f"Total masks: {len(masks)}, Masks after BBox filtering: {len(filtered_masks)}")

            # Optical Flow 계산
            if next_image is not None:
                image1_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                image2_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float()
                flow_pr, flow_magnitude_resized = compute_optical_flow(flow_model, 
                                                                       image1_tensor, image2_tensor,
                                                                       flow_args,
                                                                       device)
                
                print(f"type(flow_pr): {type(flow_pr)}, shape: {flow_pr.shape}")
                print(f"flow_magnitude_resized shape: {flow_magnitude_resized.shape}")
            else:
                print("다음 이미지가 없어 Optical Flow를 계산할 수 없습니다.")
                flow_magnitude_resized = None

            # Optical Flow 기반 마스크 필터링
            if flow_magnitude_resized is not None:
                threshold_flow_filter = 1.0
                final_filtered_masks = filter_masks_by_avg_flow(filtered_masks, flow_magnitude_resized, threshold=threshold_flow_filter)
            else:
                final_filtered_masks = []
                print("Optical Flow가 존재하지 않아 모든 마스크를 제거합니다.")
                
            # --- 각 마스크의 평균 Flow Magnitude 계산 ---
            mask_flow_magnitudes = []
            # for mask in final_filtered_masks:
            for mask in filtered_masks:
                # 마스크 영역의 Flow Magnitude 추출
                flow_in_mask = flow_magnitude_resized[mask['segmentation'] > 0]
                if flow_in_mask.size > 0:
                    avg_flow = np.mean(flow_in_mask)
                else:
                    avg_flow = 0.0
                mask_flow_magnitudes.append(avg_flow)
            print(f"mask_flow_magnitudes : {mask_flow_magnitudes}")
            
            # --- Bounding Box 영역의 평균 Flow Magnitude 계산 ---
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                # BBox 영역의 Flow Magnitude 추출
                flow_in_bbox = flow_magnitude_resized[y_min:y_max, x_min:x_max]
                if flow_in_bbox.size > 0:
                    avg_flow_bbox = np.mean(flow_in_bbox)
                else:
                    avg_flow_bbox = 0.0
            else:
                avg_flow_bbox = None
            
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
                        mask['avg_flow'] = mask_flow_magnitudes[idx]
                    visualize_and_save(
                        image=image,
                        masks=filtered_masks,
                        mask_flow_magnitudes=mask_flow_magnitudes,
                        bbox=bbox,
                        avg_flow_bbox=avg_flow_bbox,
                        save_path=pre_flow_save_path,
                        title="Filtered Masks Before Flow Filtering"
                    )
                
                # Optical Flow 적용 후의 마스크 시각화 및 저장
                if post_flow_dir:
                    post_flow_save_path = os.path.join(post_flow_dir, f"{name}_final_filtered_masks{ext}")
                    # final_filtered_masks에 'avg_flow' 추가 (이미 추가됨 in filter_masks_by_avg_flow)
                    final_flow_magnitudes = [mask['avg_flow'] for mask in final_filtered_masks]
                    visualize_and_save(
                        image=image,
                        masks=final_filtered_masks,
                        mask_flow_magnitudes=final_flow_magnitudes,
                        bbox=bbox,
                        avg_flow_bbox=avg_flow_bbox,
                        save_path=post_flow_save_path,
                        title="Final Filtered Masks After Flow Filtering"
                    )
            else:
                # 시각화만 표시
                # Optical Flow 적용 전의 마스크 시각화
                plt.figure(figsize=(10,10))
                image_with_bboxes = image.copy()
                if bbox is not None:
                    cv2.rectangle(image_with_bboxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                plt.imshow(image_with_bboxes)
                show_anns(filtered_masks, alpha=0.5)
                
                ax = plt.gca()

                # 각 마스크에 평균 Flow Magnitude 표시
                for idx, mask in enumerate(filtered_masks):
                    avg_flow = mask_flow_magnitudes[idx]
                    mask_mask = mask['segmentation']
                    ys, xs = np.nonzero(mask_mask)
                    if len(xs) > 0 and len(ys) > 0:
                        min_x = xs.min()
                        min_y = ys.min()
                        # 텍스트 표시
                        ax.text(min_x + 5, min_y + 5, f"{avg_flow:.2f}", color='white', fontsize=8, weight='bold',
                                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

                # BBox의 평균 Flow Magnitude 표시
                if avg_flow_bbox is not None:
                    # BBox의 상단 좌표 계산
                    bbox_min_x = x_min
                    bbox_min_y = y_min
                    # 텍스트 표시
                    ax.text(bbox_min_x + 5, bbox_min_y + 5, f"BBox Avg Flow: {avg_flow_bbox:.2f}", color='yellow', fontsize=12, weight='bold',
                            ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

                plt.axis('off')
                plt.title("Filtered Masks Before Flow Filtering")
                plt.show()
                plt.close()

                # Optical Flow 적용 후의 마스크 시각화
                if final_filtered_masks:
                    plt.figure(figsize=(10,10))
                    image_with_bboxes = image.copy()
                    if bbox is not None:
                        cv2.rectangle(image_with_bboxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    plt.imshow(image_with_bboxes)
                    show_anns(final_filtered_masks, alpha=0.5)
                    
                    ax = plt.gca()

                    # 각 마스크에 평균 Flow Magnitude 표시
                    for mask in final_filtered_masks:
                        avg_flow = mask.get('avg_flow', 0.0)
                        mask_mask = mask['segmentation']
                        ys, xs = np.nonzero(mask_mask)
                        if len(xs) > 0 and len(ys) > 0:
                            min_x = xs.min()
                            min_y = ys.min()
                            # 텍스트 표시
                            ax.text(min_x + 5, min_y + 5, f"{avg_flow:.2f}", color='white', fontsize=8, weight='bold',
                                    ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

                    # BBox의 평균 Flow Magnitude 표시
                    if avg_flow_bbox is not None:
                        # BBox의 상단 좌표 계산
                        bbox_min_x = x_min
                        bbox_min_y = y_min
                        # 텍스트 표시
                        ax.text(bbox_min_x + 5, bbox_min_y + 5, f"BBox Avg Flow: {avg_flow_bbox:.2f}", color='yellow', fontsize=12, weight='bold',
                                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

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

        except Exception as e:
            print(f"Error processing {path}: {e}")
            logger.error(f"Error processing {path}: {e}")
            continue  # 다음 이미지로 넘어감

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
    parser.add_argument("--flow_checkpoint", default="unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", type=str, help="UniMatch Optical Flow model checkpoint path")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)

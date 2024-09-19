# https://github.com/Kyyle2114/SAM-Inference 참고

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "unimatch"))

import cv2
import argparse
from glob import glob

from transformers import pipeline
from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

from unimatch.unimatch import UniMatch
# from evaluate_flow_colpred import inference_flow
import torch.nn.functional as F

# initialize global varaibles
drawing = False  # True면 마우스를 누르고 있는 상태
ix, iy = -1, -1  # 시작 좌표
bbox = None  # bounding box 저장 변수
points = []
labels = []
point_interval = 50

# bounding box 중심 좌표 계산 함수


def get_center_of_bbox(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, resized_image

    # 마우스를 눌렀을 때 시작 좌표 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 마우스를 이동할 때 (그리고 있는 상태에서)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = resized_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    # 마우스를 뗐을 때, 끝 좌표 저장 및 사각형 그리기
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x - ix, y - iy)
        cv2.rectangle(resized_image, (ix, iy), (x, y), (0, 255, 0), 2)

        # 중심 좌표 계산 및 출력
        center = get_center_of_bbox(ix, iy, x - ix, y - iy)
        print(f"Center of bounding box: {center}")

        # 중심점에서 y축 아래로 20픽셀 간격으로 좌표를 생성
        # points = [(center[0], center[1] + i * point_interval) for i in range(1, 11)]
        image_height = resized_image.shape[0]
        # points = []
        for i in range(1, image_height):
            new_y = center[1] + i * point_interval
            if new_y >= image_height:
                break
            points.append((center[0], new_y))
            labels.append(1)

        # 중심점 표시 (optional)
        # cv2.circle(resized_image, center, 5, (0, 0, 255), -1)
        for point in points:
            # print(f"Point: {point}")
            cv2.circle(resized_image, point, 5, (255, 0, 0), -1)
        cv2.imshow('Image', resized_image)

# SAM utils


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.0)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.0)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


def reset_bbox():
    global ix, iy, bbox, points, labels, resized_image, resized_image_copy
    ix, iy = -1, -1
    bbox = None
    points.clear()
    labels.clear()
    resized_image = resized_image_copy.copy()
    cv2.imshow("Image", resized_image)
    print("Bounding box and points have been reset.")


def process_image(img_path, output_folder, predictor, pipe, mask_index, save_all_masks, point_interval=50):
    global points, labels, resized_image, resized_image_copy
    # image load
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, (1280, 720))  # 이미지가 너무 커서 사이즈 조절
    resized_image_copy = resized_image.copy()

    # SAM predictor set image
    predictor.set_image(resized_image)

    # depth image
    pil_image = Image.fromarray(resized_image)
    depth_array = pipe(pil_image)["depth"]
    depth_nparray = np.array(depth_array)
    depth = cv2.cvtColor(depth_nparray, cv2.COLOR_RGB2BGR)

    # depth save
    base_filename = os.path.basename(img_path)
    base_filename_wo_ext, _ = os.path.splitext(base_filename)
    depth_result_path = os.path.join(
        output_folder, f"{base_filename_wo_ext}_depth.jpg")
    cv2.imwrite(depth_result_path, depth)

    cv2.imshow('Image', resized_image)
    # cv2.imshow('Depth', depth)
    cv2.setMouseCallback('Image', draw_rectangle)

    print('press the keyboard')
    print('i: Model Inference / q: Quit / r: Reset Bbox\n')

    while True:
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            points = []
            resized_image = resized_image_copy.copy()
            cv2.imshow('Image', resized_image)
            break

        elif key & 0xFF == ord('r'):
            reset_bbox()

        elif key & 0xFF == ord('i'):
            print('Now: Model Inference')
            points_sam = np.array(points)
            labels_sam = np.array(labels)

            if points_sam.size == 0:
                print('Error: No Input Points')
                break

            masks, scores, logits = predictor.predict(
                point_coords=points_sam,
                point_labels=labels_sam,
                box=None,
                multimask_output=True,
            )

            # 마스크를 적용한 이미지를 생성
            # mask_overlay = resized_image.copy()
            # mask = masks[0]
            # mask = (mask > 0).astype(np.uint8) * 255
            # # color mask
            # color_mask = np.zeros_like(resized_image)
            # color_mask[mask == 255] = (255, 0, 0)  # blue

            # # alpha blending
            # alpha = 0.5
            # blended_image = cv2.addWeighted(
            #     resized_image, 1-alpha, color_mask, alpha, 0)

            if save_all_masks:
                for mask_index in range(3):  # 0, 1, 2에 대해 모두 저장
                    plt.figure(
                        num=f"Segmentation Result - Mask {mask_index}", figsize=(8, 5))
                    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                    show_mask(masks[mask_index], plt.gca())

                    if points_sam.size != 0:
                        show_points(points_sam, labels_sam, plt.gca())

                    plt.axis('off')
                    result_path = os.path.join(
                        output_folder, f"{base_filename_wo_ext}_sam_{mask_index}.jpg")
                    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(
                        f"Saved result for {img_path} with mask {mask_index} to {result_path}")
            else:
                plt.figure(num="Segmentation Result", figsize=(10, 5))
                plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                show_mask(masks[mask_index], plt.gca())

                if points_sam.size != 0:
                    show_points(points_sam, labels_sam, plt.gca())

                plt.axis('off')

                # 결과를 저장할 경로 생성 및 저장
                result_path = os.path.join(
                    output_folder, f"{base_filename_wo_ext}_sam_{mask_index}.jpg")
                plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
                plt.show()
                plt.close()
                # cv2.imwrite(result_path, blended_image)

                print(f"Saved result for {img_path} to {result_path}")
                # break

    cv2.destroyAllWindows()

# Optical flow

def get_largest_flow_points(flow_map, top_k=1):
    """Optical flow 결과에서 가장 큰 움직임을 가진 포인트들을 추출합니다."""
    flow_magnitude = np.sqrt(flow_map[..., 0] ** 2 + flow_map[..., 1] ** 2)
    largest_indices = np.argpartition(flow_magnitude.flatten(), -top_k)[-top_k:]
    largest_points = np.unravel_index(largest_indices, flow_magnitude.shape)
    return list(zip(largest_points[1], largest_points[0]))  # (x, y) 좌표로 변환

def get_largest_flow_points_2(flow_map, top_k=1, min_distance_ratio=0.05):
    """
    Optical flow 결과에서 가장 큰 움직임을 가진 포인트들을 추출하며,
    비슷한 좌표(거리가 가까운 좌표)를 제외합니다. min_distance는 이미지 크기에 대한 비율입니다.

    Args:
        flow_map (numpy.ndarray): Optical flow 맵. [H, W, 2] 형식.
        top_k (int): 추출할 가장 큰 움직임을 가진 포인트의 개수.
        min_distance_ratio (float): 비슷한 좌표를 제외하기 위한 최소 거리 비율 (이미지 크기 대비).
    
    Returns:
        points (list): 비슷한 좌표를 제외한, 움직임이 가장 큰 top_k 포인트의 좌표.
    """
    # Flow 크기 계산
    flow_magnitude = np.sqrt(flow_map[..., 0] ** 2 + flow_map[..., 1] ** 2)
    
    # 이미지 크기
    height, width = flow_map.shape[:2]
    
    # 최소 거리를 이미지 크기 비율로 계산 (폭과 높이를 모두 고려)
    min_distance = min(width, height) * min_distance_ratio
    
    # 움직임 크기를 기준으로 내림차순으로 좌표 정렬
    flat_indices = np.argsort(flow_magnitude.flatten())[::-1]
    sorted_points = np.unravel_index(flat_indices, flow_magnitude.shape)
    points = list(zip(sorted_points[1], sorted_points[0]))  # (x, y) 좌표로 변환
    
    # 비슷한 좌표(거리가 가까운 좌표) 제외
    selected_points = []
    
    for point in points:
        if not selected_points:
            selected_points.append(point)
        else:
            # 기존에 선택된 좌표들과 비교하여 min_distance보다 큰 좌표만 추가
            if all(np.linalg.norm(np.array(point) - np.array(selected_point)) >= min_distance for selected_point in selected_points):
                selected_points.append(point)
        
        # top_k에 도달하면 멈춤
        if len(selected_points) == top_k:
            break
    
    # 선택된 좌표 반환
    return selected_points


def resize_image(image1, image2, padding_factor, inference_size=None):
    """이미지를 지정된 padding_factor로 리사이즈하고, inference_size에 맞춰 조정합니다."""
    nearest_size = [
        int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor
    ]

    inference_size = nearest_size if inference_size is None else inference_size
    ori_size = image1.shape[-2:]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

    return image1, image2, ori_size


def prepare_images(image1_path, image2_path):
    """이미지 로드 및 전처리: 이미지를 로드하고, RGB로 변환하고, 텐서로 변환합니다."""
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print(f"Error: Failed to load images {image1_path} or {image2_path}")
        return None, None

    image1 = np.array(image1).astype(np.uint8)
    image2 = np.array(image2).astype(np.uint8)

    if len(image1.shape) == 2:
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0)

    return image1, image2


def process_image_with_flow(img_dir, output_folder, predictor, pipe, mask_index, save_all_masks, flow_model, flow_args, top_k):
    """이미지 쌍을 처리하여 optical flow를 계산하고, SAM 모델을 이용한 결과를 저장합니다."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_model.eval()

    image_files = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))
    if len(image_files) < 2:
        print(f"Error: At least two images are required in the directory: {img_dir}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(image_files) - 1):
        image1_path, image2_path = image_files[i], image_files[i + 1]
        print(f"Processing images: {image1_path} and {image2_path}")

        image1, image2 = prepare_images(image1_path, image2_path)
        if image1 is None or image2 is None:
            continue

        image1 = image1.to(device)
        image2 = image2.to(device)

        # 이미지 전치 (width > height)
        transpose_img = False
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        # 리사이즈
        image1, image2, ori_size = resize_image(image1, image2, flow_args.padding_factor, flow_args.inference_size)

        if flow_args.pred_bwd_flow:
            image1, image2 = image2, image1

        # Optical Flow 계산
        results_dict = flow_model(image1, image2, attn_type=flow_args.attn_type,
                                  attn_splits_list=flow_args.attn_splits_list,
                                  corr_radius_list=flow_args.corr_radius_list,
                                  prop_radius_list=flow_args.prop_radius_list,
                                  num_reg_refine=flow_args.num_reg_refine, task='flow')

        flow_pr = results_dict['flow_preds'][-1][0].permute(1, 2, 0).detach().cpu().numpy()

        # 가장 큰 optical flow 포인트 추출
        # points = get_largest_flow_points(flow_pr, top_k=top_k) # top_k로 조절
        points = get_largest_flow_points_2(flow_pr, top_k=top_k) # top_k로 조절
        labels = [1] * len(points)

        # SAM 모델에 이미지 설정
        image1_np = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        predictor.set_image(image1_np)
        
        points_sam = np.array(points)
        labels_sam = np.array(labels)

        if points_sam.size == 0:
            print('Error: No input points')
            continue
        
        print(f"points_sam: {points_sam}, points_sam.shape: {points_sam.shape}")
        print(f"labels_sam: {labels_sam}")
        
        # SAM Inference
        masks, scores, logits = predictor.predict(point_coords=points_sam, point_labels=labels_sam, box=None, multimask_output=False)
        print(f"main.py, masks.shape: {masks.shape}")
        if len(masks.shape) >= 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        print(f"main.py, after squeeze, masks.shape: {masks.shape}")
        # 결과 저장
        save_sam_results(masks, points_sam, labels_sam, image1_np, output_folder, image1_path, mask_index, save_all_masks)

    print("Processing completed.")


def save_sam_results(masks, points_sam, labels_sam, image, output_folder, image_path, mask_index, save_all_masks):
    """SAM 결과를 저장하는 함수."""
    base_filename = os.path.basename(image_path)
    base_filename_wo_ext, _ = os.path.splitext(base_filename)

    if save_all_masks:
        for idx in range(masks.shape[0]):
            save_single_sam_result(masks, points_sam, labels_sam, image, output_folder, base_filename_wo_ext, idx)
    else:
        save_single_sam_result(masks, points_sam, labels_sam, image, output_folder, base_filename_wo_ext, mask_index)


def save_single_sam_result(masks, points_sam, labels_sam, image, output_folder, base_filename_wo_ext, idx):
    """단일 SAM 마스크 결과 저장."""
    plt.figure(num=f"Segmentation Result - Mask {idx}", figsize=(8, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        show_mask(masks[idx], plt.gca())
    except IndexError:
        show_mask(masks, plt.gca())
    show_points(points_sam, labels_sam, plt.gca())
    plt.axis('off')

    result_path = os.path.join(output_folder, f"{base_filename_wo_ext}_sam_{idx}.jpg")
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved result to {result_path}")


def main(args):
    # SAM 모델 로드

    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # SAM 모델 타입에 따른 체크포인트 설정
    if args.model_type == "vit_h":
        sam_checkpoint = "sam_vit_h_4b8939.pth"
    elif args.model_type == "vit_l":
        sam_checkpoint = "sam_vit_l_0b3195.pth"
    elif args.model_type == "vit_b":
        sam_checkpoint = "sam_vit_b_01ec64.pth"
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    print(f'Loading Segment Anything Model: {args.model_type}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Depth estimation pipeline 로드
    pipe = pipeline(task='depth-estimation',
                    model='depth-anything/Depth-Anything-V2-Small-hf', device=0)

    # 출력 폴더가 없으면 생성
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    # Option 1. bounding box of hoist -> point prompt -> sam
    # process_image(args.input, args.output, predictor, pipe, args.mask_index, args.save_all_masks)
    
    
    # Option 2. optical flow model
    
    # Optical Flow 모델 빌드 (main_flow.py에서 가져온 코드)
    print('Loading UniMatch Optical Flow Model')
    flow_model = UniMatch(
        feature_channels=128,  # default: 128
        num_scales=2,  # default: 1 # demo: 2
        upsample_factor=4,  # default: 8 # demo: 4
        num_head=1,  # default: 1
        ffn_dim_expansion=4,  # default: 4
        num_transformer_layers=6,  # default: 6
        reg_refine=True,  # demo: active 
        task='flow'  # flow task 지정
    ).to(device)
    
    # model weight file path
    flow_model_checkpoint = "unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"  # Optical Flow 모델 경로

    # load weights
    if flow_model_checkpoint is not None:
        print(f"Loading checkpoint from {flow_model_checkpoint}")
        checkpoint = torch.load(flow_model_checkpoint, map_location=device)
        flow_model.load_state_dict(checkpoint['model'], strict=False)
        print("Model weights loaded successfully.")
    
    # Optical Flow 추론을 위한 arguments 설정
    flow_args = argparse.Namespace(
        inference_dir=args.input,  # 입력 디렉토리 (이미지 경로)
        output_path=args.output,   # 결과 저장 디렉토리
        padding_factor=32,         # 패딩 팩터 설정
        inference_size=None, # 추론 크기
        pred_bwd_flow=False,
        attn_type='swin',
        attn_splits_list=[2, 8],   # attention split list
        corr_radius_list=[-1, 4],  # correlation 반경 리스트
        prop_radius_list=[-1, 1],  # propagation 반경 리스트
        num_reg_refine=6           # refinement 스텝 개수
    )
    
    process_image_with_flow(
        img_dir=args.input,
        output_folder=args.output,
        predictor=predictor,
        pipe=pipe,
        mask_index=args.mask_index,
        save_all_masks=args.save_all_masks,
        flow_model=flow_model,
        flow_args=flow_args,
        top_k=args.top_k
    )
    print("Processing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM Inference Script")
    parser.add_argument('--input', type=str, required=True,
                        help="Input image path")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument('--model_type', type=str, default="vit_h", choices=[
                        "vit_h", "vit_l", "vit_b"], help="Model type for SAM (vit_h, vit_l, vit_b)")
    parser.add_argument('--mask_index', type=int, default=2,
                        choices=[0, 1, 2], help="Index for mask selection (0, 1, 2)")
    parser.add_argument('--save_all_masks', action='store_true',
                        help="Save results for all mask indices (0, 1, 2)")
    parser.add_argument('--top_k', type=int, default=5, help='top k points of optical flow for masks')
    args = parser.parse_args()

    main(args)

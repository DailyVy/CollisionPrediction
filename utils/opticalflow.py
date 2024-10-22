import os

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image

from unimatch.unimatch.unimatch import UniMatch
from unimatch.utils.flow_viz import *

def load_unimatch_model(checkpoint_path, device="cuda"):
    """UniMatch Optical Flow 모델 로드 및 초기화."""
    print('Loading UniMatch Optical Flow Model')
    flow_model = UniMatch(
        feature_channels=128,  # default: 128
        num_scales=2,  # default: 1 # demo: 2
        upsample_factor=4,  # default: 8 # demo: 4
        num_head=1,  # default: 1
        ffn_dim_expansion=4,  # default: 4
        num_transformer_layers=6,  # default: 6
        reg_refine=True,  # demo: active 
    ).to(device)
    
    # model weight file path
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['model'], strict=False)
        print("Model weights loaded successfully.")
    
    flow_model.eval()
    return flow_model

@torch.no_grad()
def compute_optical_flow(flow_model, image1, image2, flow_args, device):
    """
    UniMatch를 사용하여 Optical Flow를 계산합니다.

    Args:
        flow_model (UniMatch): UniMatch Optical Flow 모델.
        image1 (torch.Tensor): 첫 번째 이미지 텐서.
        image2 (torch.Tensor): 두 번째 이미지 텐서.
        flow_args (argparse.Namespace): Optical Flow 계산에 필요한 추가 인자.

    Returns:
        flow_pr (np.ndarray): Optical Flow 예측 결과.
        ori_size (tuple): 원본 이미지의 (높이, 너비).
    """
    # 이미지 전처리
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    image1_resized, image2_resized, ori_size, inference_size = resize_image(image1, image2, flow_args.padding_factor, flow_args.task)
    
    # Optical Flow 계산
    results_dict = flow_model(
        image1_resized, image2_resized, 
        attn_type=flow_args.attn_type,
        attn_splits_list=flow_args.attn_splits_list,
        corr_radius_list=flow_args.corr_radius_list,
        prop_radius_list=flow_args.prop_radius_list,
        num_reg_refine=flow_args.num_reg_refine, 
        task=flow_args.task
    )
    
    # flow_pr = results_dict['flow_preds'][-1][0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    flow_pr = results_dict['flow_preds'][-1]
    
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[1] / inference_size[1]  # x축 스케일링
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[0] / inference_size[0]  # y축 스케일링

        # Flow 벡터 조정 후, Numpy 배열로 변환
        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        # Flow Magnitude 계산 및 리사이즈
        flow_magnitude = np.linalg.norm(flow, axis=2)
        flow_magnitude_resized = resize_flow_magnitude(flow_magnitude, ori_size)

        return flow, flow_magnitude_resized
    
    return flow_pr[0].permute(1, 2, 0).cpu().numpy(), ori_size

def resize_image(image1, image2, padding_factor, task):
    """이미지를 지정된 padding_factor로 리사이즈하고, inference_size에 맞춰 조정합니다."""
    # smaller inference size for faster speed
    max_inference_size = [384, 768] if task == 'flow' else [640, 960]
    
    nearest_size = [
        int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor
    ]

    inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]
    ori_size = image1.shape[-2:]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

    return image1, image2, ori_size, inference_size

def optical_flow_uni_match(flow_model, image1, image2, flow_args):
    """UniMatch를 사용하여 Optical Flow 계산."""
    # 이미지 전처리 (예시, 실제 구현 필요)
    # 이미지가 tensor 형태라면 다음과 같이 변환
    # image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
    # image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0
    
    # Optical Flow 계산 (UniMatch의 forward 메소드에 따라 조정 필요)
    
    image1, image2, ori_size, inference_size = resize_image(image1, image2, flow_args.padding_factor, flow_args.task)
    with torch.no_grad():
        results_dict = flow_model(image1, image2, attn_type=flow_args.attn_type,
                                  attn_splits_list=flow_args.attn_splits_list,
                                  corr_radius_list=flow_args.corr_radius_list,
                                  prop_radius_list=flow_args.prop_radius_list,
                                  num_reg_refine=flow_args.num_reg_refine, task=flow_args.task)
        flow_pr = results_dict['flow_preds'][-1][0].permute(1, 2, 0).cpu().numpy()

    return flow_pr, ori_size

def calculate_flow_magnitude(flow_pr):
    """
    Optical Flow 결과의 Magnitude를 계산합니다.

    Args:
        flow_pr (np.ndarray): Optical Flow 예측 결과 [H, W, 2].

    Returns:
        flow_magnitude (np.ndarray): Flow Magnitude 배열 [H, W].
    """
    flow_magnitude = np.linalg.norm(flow_pr, axis=2)
    return flow_magnitude

def resize_flow_magnitude(flow_magnitude, ori_size):
    """
    Flow Magnitude를 원본 이미지 크기로 리사이즈합니다.

    Args:
        flow_magnitude (np.ndarray): Flow Magnitude 배열 [H, W].
        ori_size (tuple): 원본 이미지의 (높이, 너비).

    Returns:
        flow_magnitude_resized (np.ndarray): 리사이즈된 Flow Magnitude 배열 [H, W].
    """
    flow_magnitude_resized = cv2.resize(flow_magnitude, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_LINEAR)
    return flow_magnitude_resized

def visualize_flow(flow_pr, save_path=None):
    """
    Optical Flow를 시각화합니다.

    Args:
        flow_pr (np.ndarray): Optical Flow 예측 결과 [H, W, 2].
        save_path (str, optional): 저장할 파일 경로.

    Returns:
        flow_vis (np.ndarray): 시각화된 Flow 이미지.
    """
    flow_vis = flow_to_image(flow_pr)  # [H, W, 3]
    vis_image = Image.fromarray(flow_vis)
    if save_path:
        vis_image.save(os.path.join("./Output_OF", save_path),'JPEG')
    return flow_vis

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

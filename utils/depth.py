# utils/depth.py

import numpy as np
import torch
import cv2

def mask_has_similar_depth(mask, depth_mag, bbox_depth, threshold=0.1):
    """
    마스크의 평균 Depth가 Bounding Box의 Depth와 유사한지 확인합니다.

    Args:
        mask (np.ndarray): 2D 이진 마스크 배열.
        depth_mag (np.ndarray): Depth 맵 배열.
        bbox_depth (float): Bounding Box의 평균 Depth.
        threshold (float): Depth 유사성 임계값 (비율, 예: 0.1은 ±10%).

    Returns:
        bool: 마스크의 평균 Depth가 임계값 내에 있으면 True, 아니면 False.
    """
    if depth_mag is None:
        return False
    # 마스크 영역 내의 Depth 추출
    depth_in_mask = depth_mag[mask > 0]
    if depth_in_mask.size == 0:
        return False
    # 마스크의 평균 Depth 계산
    avg_depth = np.mean(depth_in_mask)
    # Bounding Box Depth와 유사한지 확인
    lower_bound = bbox_depth * (1 - threshold)
    upper_bound = bbox_depth * (1 + threshold)
    return lower_bound <= avg_depth <= upper_bound

def filter_masks_by_depth(masks, depth_mag, bbox_depth, threshold=0.1):
    """
    마스크의 평균 Depth를 기준으로 마스크를 필터링합니다.

    Args:
        masks (list): 마스크 리스트.
        depth_mag (np.ndarray): Depth 맵 배열.
        bbox_depth (float): Bounding Box의 평균 Depth.
        threshold (float): Depth 유사성 임계값 (비율).

    Returns:
        list: Depth 유사성이 높은 마스크 리스트.
    """
    final_filtered_masks = []
    for mask in masks:
        if mask_has_similar_depth(mask['segmentation'], depth_mag, bbox_depth, threshold):
            # 마스크 딕셔너리에 'avg_depth' 키 추가
            depth_in_mask = depth_mag[mask['segmentation'] > 0]
            avg_depth = np.mean(depth_in_mask) if depth_in_mask.size > 0 else 0.0
            mask['avg_depth'] = avg_depth
            final_filtered_masks.append(mask)
    print(f"Depth 유사성을 고려한 최종 마스크 수: {len(final_filtered_masks)} (Threshold: {threshold})")
    return final_filtered_masks

def normalize_depth_to_meters(depth_value, min_depth=0.1, max_depth=10.0):
    """
    min_depth: 최소 거리 (미터)
    max_depth: 최대 거리 (미터)
    """
    
    print(f"depth_value : {depth_value}")
    # 0-1 범위로 정규화
    normalized = 1.0 - (depth_value / 255.0)
    
    # 0-1 범위를 실제 거리(미터)로 변환
    depth_meters = normalized * (max_depth - min_depth) + min_depth
    
    return depth_meters

def get_depth_at_position(position, depth_map):
    """
    주어진 위치의 depth 값을 미터 단위로 반환
    """
    x, y = map(int, position)
    raw_depth = depth_map[y, x]
    return normalize_depth_to_meters(raw_depth)

def get_3d_positions(objects, depth_map):
    """
    객체들의 3D 위치 계산 (z축은 미터 단위)
    """
    if not objects:
        return []
    
    positions_3d = []
    for obj in objects:
        x, y = obj['position']
        depth_meters = get_depth_at_position((x, y), depth_map)
        
        positions_3d.append({
            'position': obj['position'],
            'bbox': obj.get('bbox', None),
            'confidence': obj.get('confidence', None),
            'position_3d': (x, y, depth_meters)
        })
    
    return positions_3d

def process_depth(image, model, processor):
    """배치 처리를 위한 depth estimation 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 예측된 depth를 numpy 배열로 변환
    depth_map = predicted_depth.squeeze().cpu().numpy()
    
    # Min-max scaling to 0-255 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    
    # 원본 이미지 크기로 리사이즈
    h, w = image.shape[:2]  # 원본 이미지 크기
    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth_map
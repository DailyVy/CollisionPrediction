# utils/depth.py

import numpy as np

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

import numpy as np

def mask_overlaps_bbox_x(mask, x_min, x_max):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False

    mask_x_min = xs.min()
    mask_x_max = xs.max()

    return not (mask_x_max < x_min or mask_x_min > x_max)

def get_mask_position(mask):
    if not isinstance(mask['segmentation'], list):
        mask_points = np.where(mask['segmentation'])
        x_coords = mask_points[1] 
        y_coords = mask_points[0] 
    else:
        coords = np.array(mask['segmentation'][0])
        x_coords = coords[::2]  
        y_coords = coords[1::2]  
    
    x_median = np.median(x_coords)
    y_median = np.median(y_coords)

    return x_median, y_median 

def merge_overlapping_masks(masks):
    """여러 개의 마스크를 하나로 합칩니다."""
    if not masks:
        return []
    
    # 모든 마스크의 segmentation을 하나의 배열로 합침
    merged_mask = np.zeros_like(masks[0]['segmentation'], dtype=bool)
    for mask in masks:
        merged_mask = np.logical_or(merged_mask, mask['segmentation'])
    
    # 새로운 마스크 딕셔너리 생성
    merged_mask_dict = {
        'segmentation': merged_mask,
        'area': float(np.sum(merged_mask)),  # 합쳐진 마스크의 면적
        'bbox': masks[0]['bbox'],  # 첫 번째 마스크의 bbox 사용
        'predicted_iou': max(mask['predicted_iou'] for mask in masks),  # 가장 높은 IoU 사용
        'point_coords': masks[0]['point_coords'],  # 첫 번째 마스크의 좌표 사용
        'stability_score': max(mask['stability_score'] for mask in masks),  # 가장 높은 안정성 점수 사용
        'crop_box': masks[0]['crop_box']  # 첫 번째 마스크의 crop_box 사용
    }
    
    return [merged_mask_dict]
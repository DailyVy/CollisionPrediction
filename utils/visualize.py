import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Dict, Any
from utils.mask_utils import get_mask_position
def show_anns(anns, alpha=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)

def flow_visualize_and_save(image, masks, mask_flow_magnitudes, bbox, avg_flow_bbox, save_path, title="Masks Visualization"):
    """
    마스크를 시각화하고 이미지를 저장합니다.

    Args:
        image (np.ndarray): 원본 이미지 (RGB).
        masks (list): 마스크 리스트. 각 마스크는 'segmentation'과 'avg_flow' 키를 포함해야 함.
        mask_flow_magnitudes (list): 각 마스크의 평균 Flow Magnitude 리스트.
        bbox (tuple or None): Bounding Box 좌표 (x_min, y_min, x_max, y_max).
        avg_flow_bbox (float or None): BBox의 평균 Flow Magnitude.
        save_path (str): 저장할 파일 경로.
        title (str): 플롯의 제목.
    """
    plt.figure(figsize=(10,10))
    image_with_bboxes = image.copy()
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_with_bboxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    plt.imshow(image_with_bboxes)
    show_anns(masks, alpha=0.5)

    ax = plt.gca()

    # 각 마스크에 평균 Flow Magnitude 표시
    for idx, mask in enumerate(masks):
        avg_flow = mask_flow_magnitudes[idx]
        mask_mask = mask['segmentation']
        ys, xs = np.nonzero(mask_mask)
        if len(xs) > 0 and len(ys) > 0:
            min_x = xs.min()
            min_y = ys.min()
            # 텍스트 표시 (왼쪽 상단에 약간의 오프셋 추가)
            ax.text(min_x + 5, min_y + 5, f"{avg_flow:.2f}", color='white', fontsize=8, weight='bold',
                    ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # BBox의 평균 Flow Magnitude 표시
    if bbox is not None and avg_flow_bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        # 텍스트 표시 (왼쪽 상단에 약간의 오프셋 추가)
        ax.text(x_min + 5, y_min + 5, f"BBox Avg Flow: {avg_flow_bbox:.2f}", color='yellow', fontsize=12, weight='bold',
                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to: {save_path}")
    
def mask_visualize_and_save(image, masks, save_path, title="Masks Visualization"):
    """
    마스크를 시각화하고 이미지를 저장합니다.

    Args:
        image (np.ndarray): 원본 이미지 (RGB).
        masks (list): 마스크 리스트. 각 마스크는 'segmentation' 키를 포함해야 함.
        save_path (str): 저장할 파일 경로.
        title (str): 플롯의 제목.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    
    # 마스크 시각화
    show_anns(masks, alpha=0.5)

    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to: {save_path}")

def depth_visualize_and_save(image, final_masks, masks, mask_depth_magnitudes, bbox, avg_depth_bbox, save_path, title="Depth Filtered Masks"):
    """
    마스크와 관련 정보를 이미지에 시각화하고 저장하는 함수.

    Args:
        image (np.ndarray): RGB 이미지 배열.
        masks (list): 마스크 리스트.
        mask_depth_magnitudes (list): 각 마스크의 평균 Depth 값 리스트.
        bbox (tuple or None): Bounding Box 좌표 (x_min, y_min, x_max, y_max).
        avg_depth_bbox (float or None): Bounding Box의 평균 Depth.
        save_path (str): 저장할 파일 경로.
        title (str): 이미지 제목.
    """
    plt.figure(figsize=(10, 10))
    image_with_bboxes = image.copy()
    
    # Bounding Box가 존재하면 이미지에 사각형 그리기
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_with_bboxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    plt.imshow(image_with_bboxes)
    show_anns(final_masks, alpha=0.5)  # 필터링된 마스크만 오버레이
    
    ax = plt.gca()

    # 모든 마스크의 Depth 정보 표시
    for idx, (mask, avg_depth) in enumerate(zip(masks, mask_depth_magnitudes)):
        mask_mask = mask['segmentation']
        ys, xs = np.nonzero(mask_mask)
        if len(xs) > 0 and len(ys) > 0:
            min_x = xs.min()
            min_y = ys.min()
            # Depth 정보 텍스트 표시
            ax.text(min_x + 5, min_y + 5, f"Depth: {avg_depth:.2f}", color='white', fontsize=8, weight='bold',
                    ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # Bounding Box의 평균 Depth 표시
    if bbox is not None and avg_depth_bbox is not None:
        bbox_min_x, bbox_min_y, _, _ = bbox
        ax.text(bbox_min_x + 5, bbox_min_y + 5, f"Hoist Avg Depth: {avg_depth_bbox:.2f}", color='yellow', fontsize=12, weight='bold',
                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.axis('off')
    plt.title(title)
    plt.tight_layout(pad=0)  # 여백 최소화
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to: {save_path}")

def visualize_with_positions(image, final_filtered_masks, person_positions, forklift_positions, title="Objects Positions"):
    """
    이미지에 마스크와 각 객체의 위치를 시각화
    
    Args:
        image: 원본 이미지
        final_filtered_masks: 필터링된 마스크들
        person_positions: 사람 위치 리스트
        forklift_positions: 지게차 위치 리스트
        title: 그래프 제목
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # 마스크 시각화
    if final_filtered_masks:
        show_anns(final_filtered_masks, alpha=0.5)
        
        # Heavy object (마스크) 중위값 위치 표시
        for idx, mask in enumerate(final_filtered_masks):
            x_pos, y_pos = get_mask_position(mask)
            # 첫 번째 마스크일 때만 레이블 추가
            plt.plot(x_pos, y_pos, 'r*', markersize=15, label='Heavy Object' if idx == 0 else "")
    # 사람 위치 표시
    for person in person_positions:
        x, y = person['position']
        plt.plot(x, y, 'go', markersize=10, label='Person' if person == person_positions[0] else "")
    
    # 지게차 위치 표시
    for forklift in forklift_positions:
        x, y = forklift['position']
        plt.plot(x, y, 'bs', markersize=10, label='Forklift' if forklift == forklift_positions[0] else "")
    
    plt.axis('off')
    plt.title(title)
    plt.legend()
    
    return plt.gcf()


def visualize_with_3d_positions(
    image: np.ndarray,
    person_positions_3d: List[Dict[str, Any]], 
    forklift_positions_3d: List[Dict[str, Any]], 
    heavy_objects_positions_3d: List[Dict[str, Any]], 
    masks: List[Dict[str, Any]], 
    title: str = "Objects 3D Positions"
) -> plt.Figure:
    """
    이미지에 마스크와 각 객체의 3D 위치를 시각화

    Args:
        image: 원본 이미지
        final_filtered_masks: 필터링된 마스크들
        person_positions_3d: 사람 3D 위치 리스트
        forklift_positions_3d: 지게차 3D 위치 리스트
        heavy_objects_positions_3d: heavy object 3D 위치 리스트
        title: 그래프 제목

    Returns:
        plt.Figure: 생성된 matplotlib Figure 객체
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # 마스크 시각화
    if masks:
        show_anns(masks, alpha=0.5)
    
    # Heavy object 위치 표시
    for idx, obj in enumerate(heavy_objects_positions_3d):
        x, y, z = obj['position_3d']
        label = f'Heavy Object (x: {x:.2f}px, y: {y:.2f}px, z: {z:.2f})'
        plt.plot(x, y, 'r*', markersize=15, label=label if idx == 0 else "")
    
    # 사람 위치 표시
    for idx, person in enumerate(person_positions_3d):
        x, y, z = person['position_3d']
        label = f'Person (x: {x:.2f}px, y: {y:.2f}px, z: {z:.2f})'
        plt.plot(x, y, 'go', markersize=10, label=label if idx == 0 else "")
    
    # 지게차 위치 표시
    for idx, forklift in enumerate(forklift_positions_3d):
        x, y, z = forklift['position_3d']
        label = f'Forklift (x: {x:.2f}px, y: {y:.2f}px, z: {z:.2f})'
        plt.plot(x, y, 'bs', markersize=10, label=label if idx == 0 else "")
    
    plt.axis('off')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return plt.gcf()

def visualize_with_distances(
    image,
    heavy_objects_positions_3d,
    person_positions_3d,
    forklift_positions_3d,
    distances,
    masks=None,
    title="3D Positions and Distances"
):
    """
    3D 위치와 객체 간 거리를 시각화합니다.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # 마스크 시각화
    if masks:
        show_anns(masks, alpha=0.5)
    
    # Heavy object 위치 표시
    for idx, obj in enumerate(heavy_objects_positions_3d):
        x, y, z = obj['position_3d']
        plt.plot(x, y, 'r*', markersize=15, label=f'Heavy Object {idx+1}')
    
    # 사람 위치 표시
    for idx, person in enumerate(person_positions_3d):
        x, y, z = person['position_3d']
        plt.plot(x, y, 'go', markersize=10, label=f'Person {idx+1}')
    
    # 지게차 위치 표시
    for idx, forklift in enumerate(forklift_positions_3d):
        x, y, z = forklift['position_3d']
        plt.plot(x, y, 'bs', markersize=10, label=f'Forklift {idx+1}')
    
    # 거리 표시
    for dist in distances:
        from_id = dist['from_id']
        to_id = dist['to_id']
        details = dist['distance_details']
        
        # 시작점과 끝점 찾기
        if 'heavy_object' in from_id:
            idx = int(from_id.split('_')[-1])
            start = heavy_objects_positions_3d[idx]['position_3d'][:2]
        elif 'person' in from_id:
            idx = int(from_id.split('_')[-1])
            start = person_positions_3d[idx]['position_3d'][:2]
        elif 'forklift' in from_id:
            idx = int(from_id.split('_')[-1])
            start = forklift_positions_3d[idx]['position_3d'][:2]
            
        if 'heavy_object' in to_id:
            idx = int(to_id.split('_')[-1])
            end = heavy_objects_positions_3d[idx]['position_3d'][:2]
        elif 'person' in to_id:
            idx = int(to_id.split('_')[-1])
            end = person_positions_3d[idx]['position_3d'][:2]
        elif 'forklift' in to_id:
            idx = int(to_id.split('_')[-1])
            end = forklift_positions_3d[idx]['position_3d'][:2]
        
        # 선 그리기
        plt.plot([start[0], end[0]], [start[1], end[1]], '-', color='red', alpha=0.3)
        
        # 거리 텍스트 표시
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        plt.annotate(
            f'{details["weighted_score"]:.1f}m',
            (mid_x, mid_y),
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            ha='center',
            va='center'
        )
    
    plt.title(title)
    plt.axis('off')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return plt.gcf()
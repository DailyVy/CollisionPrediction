import matplotlib.pyplot as plt
import cv2
import numpy as np

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

def visualize_and_save(image, masks, mask_flow_magnitudes, bbox, avg_flow_bbox, save_path, title="Masks Visualization"):
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

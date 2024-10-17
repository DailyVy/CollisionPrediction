import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "unimatch"))

import argparse
from glob import glob

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline # Grounding DINO, DepthAnythingV2
from segment_anything import sam_model_registry, SamPredictor, build_sam # SAM
from unimatch.unimatch import UniMatch # UniMatch
import tqdm

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# CLASSES
TEXT = "a forklift. a person."
# TEXT = "a chain"

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_results_gd2sam(pil_img, scores, labels, boxes, masks):
    plt.figure(figsize=(16, 10))
    
    # PIL 이미지를 numpy 배열로 변환
    pil_img_np = np.array(pil_img)
    
    # 마스크가 적용될 이미지를 복사
    masked_image = pil_img_np.copy()
    
    ax = plt.gca()
    colors = COLORS * 100

    # 마스크와 바운딩 박스를 함께 그리기
    for mask, score, label, (xmin, ymin, xmax, ymax), color in zip(masks, scores, labels, boxes, colors):
        # mask: [1, H, W] -> [H, W]로 변환
        mask = mask[0]  
        
        # 마스크를 원본 이미지 위에 투명하게 덧씌움
        for c in range(3):  # 각 채널에 대해 연산
            masked_image[:, :, c] = np.where(mask == 1, masked_image[:, :, c] * (1 - 0.5) + 0.5 * color[c] * 255, masked_image[:, :, c])

        # Bounding box 표시
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=3))
        
        # 레이블과 점수를 이미지에 표시
        label_text = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label_text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    # 최종적으로 마스크가 적용된 이미지를 보여줌
    ax.imshow(masked_image, alpha=1.0)
    
    plt.axis('off')
    plt.show()

def visualize_mask_on_image(image_np, masks):
    """
    masks를 이미지에 시각화하는 함수.
    
    Args:
        image_np (numpy.ndarray): 이미지 배열 (H, W, 3).
        masks (numpy.ndarray): 마스크 배열 (B, H, W) 형태.
    """
    color = (0, 255, 255)  # Cyan (RGB)

    # 이미지 시각화 설정
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)

    # 마스크를 이미지에 반투명하게 덧씌움
    masked_image = image_np.copy()
    for c in range(3):  # RGB 채널 각각에 대해 연산
        masked_image[:, :, c] = np.where(masks == 1, 
                                         image_np[:, :, c] * (1 - 0.5) + 0.5 * color[c], 
                                         image_np[:, :, c])

    plt.imshow(masked_image, alpha=1)
    plt.title("Mask Visualization")
    plt.axis('off')
    
    plt.show()

def grounding_dino_to_sam(dino_model, dino_processor, sam_predictor, image, device) -> dict:
    
    inputs = dino_processor(images=image, text=preprocess_caption(TEXT), return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
        
    results = dino_processor.post_process_grounded_object_detection(outputs,
                                                                    inputs.input_ids,
                                                                    box_threshold=0.3,
                                                                    text_threshold=0.3,
                                                                    target_sizes=[image.size[::-1]],
                                                                    )[0]

    first_image_np = np.array(image)
    sam_predictor.set_image(first_image_np)
    
    H, W, _ = first_image_np.shape
    boxes_xyxy = results['boxes']
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, first_image_np.shape[:2]).to(device)
    
    masks, _, _ = sam_predictor.predict_torch(point_coords=None,
                                              point_labels=None,
                                              boxes=transformed_boxes,
                                              multimask_output=False,
                                              )
    
    # For Visualization
    # plot_results_gd2sam(image, 
    #                     results['scores'].tolist(), 
    #                     results['labels'], 
    #                     results['boxes'].tolist(), 
    #                     masks.cpu().numpy())
    results['masks'] = masks.cpu().numpy()
    
    return results

def hoist_bbox_sam(first_image, sam_predictor):
    # 이미지를 NumPy 배열로 변환
    image_np = np.array(first_image)
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # bbox 시작 좌표와 종료 좌표를 저장할 변수
    bbox = []

    # 마우스 이벤트 처리 콜백 함수
    def draw_rectangle(event, x, y, flags, param):
        nonlocal bbox
        if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 경우
            bbox = [(x, y)]  # 시작 좌표 기록
        elif event == cv2.EVENT_LBUTTONUP:  # 마우스를 떼면 종료 좌표 기록
            bbox.append((x, y))
            cv2.rectangle(image_np_bgr, bbox[0], bbox[1], (0, 255, 0), 2)  # 녹색 사각형 그리기
            cv2.imshow('Image', image_np_bgr)

    # OpenCV 창 열기
    cv2.imshow('Image', image_np_bgr)
    cv2.setMouseCallback('Image', draw_rectangle)

    # 사용자가 's'를 눌러서 이미지를 종료할 때까지 대기
    while True:
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()

    # bounding box 좌표를 (xmin, ymin, xmax, ymax)로 변환
    if len(bbox) == 2:
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        box_xyxy = np.array([[xmin, ymin, xmax, ymax]])

        # SAM에 맞게 박스를 변환
        transformed_box = sam_predictor.transform.apply_boxes_torch(
            torch.as_tensor(box_xyxy, dtype=torch.float, device=sam_predictor.device),
            image_np.shape[:2]
        )

        # SAM 모델로 예측
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            multimask_output=False
        )
        # For Visualization
        # visualize_mask_on_image(image_np, masks.cpu().numpy())
        return masks.cpu().numpy()
    else:
        print("No valid bounding box was drawn.")
        return None

def distance(first_image, objects_info, hoist_mask, depth_model):
    # 'a person'에 해당하는 마스크 찾기
    person_masks = []
    for score, label, mask in zip(objects_info['scores'], objects_info['labels'], objects_info['masks']):
        if label == 'a person':
            # 마스크를 NumPy 배열로 변환
            mask_np = np.array(mask)
            # 마스크의 불필요한 차원을 제거하여 2D 배열로 변환
            mask_squeezed = np.squeeze(mask_np)
            # 만약 마스크가 여전히 2차원이 아니면, 마지막 차원을 제거
            if mask_squeezed.ndim > 2:
                mask_squeezed = mask_squeezed[..., 0]
            print(f"Processed person mask shape: {mask_squeezed.shape}")
            person_masks.append(mask_squeezed)
            
    if len(person_masks) == 0:
        # raise ValueError("No 'a person' masks found in objects_info.") # todos. ValueError 대신 return으로 수정, 추후 main에서 return 값이 None이면 for문 continue
        print("No 'person' masks found in objects_info.")
        return None  # ValueError 대신 None 반환
    
    # hoist_mask도 2D 배열로 변환
    hoist_mask_np = np.array(hoist_mask)
    hoist_mask_squeezed = np.squeeze(hoist_mask_np)
    if hoist_mask_squeezed.ndim > 2:
        hoist_mask_squeezed = hoist_mask_squeezed[..., 0]
    print(f"hoist_mask's shape: {hoist_mask_squeezed.shape}")
        
    # depth 계산
    depth_array = depth_model(first_image)["depth"]
    depth_nparray = np.array(depth_array) # (360, 640)의 ndarray
    
    # depth = cv2.cvtColor(depth_nparray, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Depth', depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # hoist_mask와 각 person_mask의 평균 계산
    def compute_mean_x(mask):
        indices = np.where(mask == 1)  # 마스크 내 유효한 영역 찾기
        if len(indices[1]) == 0:
            return None
        return np.mean(indices[1])  # x좌표의 평균 계산 (indices[1]이 x좌표)
    
    def compute_mean_depth(mask, depth_nparray):
        indices = np.where(mask == 1)
        depths = []
        
        if len(indices[0]) == 0 or len(indices[1]) == 0:
            return None
        depths = depth_nparray[indices]  # 마스크 영역의 깊이 값 가져오기
        if depths.size == 0:
            return None
        return np.mean(depths)  # 평균 깊이 반환
        
    # 각 마스크의 x좌표 평균 계산
    hoist_x_mean = compute_mean_x(hoist_mask_squeezed)
    person_x_means = [compute_mean_x(mask) for mask in person_masks]

    # 각 마스크의 평균 깊이 계산 (이상치 제거 포함)
    hoist_depth_mean = compute_mean_depth(hoist_mask_squeezed, depth_nparray)
    person_depth_means = [compute_mean_depth(mask, depth_nparray) for mask in person_masks]
    
    print(f"Hoist Depth Mean: {hoist_depth_mean}")
    for idx, depth_val in enumerate(person_depth_means):
        print(f"Person {idx+1} Depth Mean: {depth_val}")

    # 거리 계산 (예시: hoist_depth_mean과 각 person_depth_mean 간의 거리)
    # 여기서는 단순히 hoist_depth_mean과 모든 person_depth_mean의 평균 차이를 계산
    if hoist_depth_mean is None or any(d is None for d in person_depth_means):
        print("Depth 계산 중 일부 값이 누락되었습니다.")
        return None
    
    # 개별 거리 값 계산 (delta_x, delta_z)
    # delta_x : 사람과 hoist간의 x축 거리 (양수: 오른쪽, 음수: 왼쪽)
    # delta_z : 사람과 hoist간의 z축 거리 (양수: 뒤, 음수: 앞)
    distance_vals = []
    for person_x_mean, person_depth_mean in zip(person_x_means, person_depth_means):
        if person_x_mean is None or person_depth_mean is None:
            distance_vals.append((None, None))
        else:
            delta_x = person_x_mean - hoist_x_mean  # x축 거리 (양수: 오른쪽, 음수: 왼쪽)
            delta_z = person_depth_mean - hoist_depth_mean  # z축 거리 (양수: 뒤, 음수: 앞)
            distance_vals.append((delta_x, delta_z))
            
    return distance_vals


def optical_flow(first_image, second_image, objects_info, hoist_mask, device):
    # Optical Flow 모델 빌드 (main_flow.py에서 가져온 코드)
    pass

def visualize_distance(image, distance_vals, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))  # 이미지가 BGR 형식일 경우 RGB로 변환
    
    if distance_vals is not None:
        for idx, distance_val in enumerate(distance_vals):
            plt.text(10, 30 + idx * 20, f"Distance {idx+1}: {distance_val:.2f}", 
                     color='red', fontsize=12, backgroundcolor='white')
    else:
        plt.text(10, 30, "Distance: N/A", color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.title("Distance Between Person Masks and Hoist Mask")
    plt.savefig(output_path)
    plt.close()

    
def main(args):
    # args 정리
    # input arguments
    img_dir = args.input
    
    # create output folders if not exists.
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")
    
    # image file 정리
    image_files = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))
    
    # ** Loading Grounding DINO model
    print('Loading Grounding DINO')
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # ** SAM load
    sam_checkpoint = "sam_vit_h_4b8939.pth" # "sam_vit_l_0b3195.pth", "sam_vit_b_01ec64.pth"
    sam_model_type = "vit_h"

    print(f'Loading Segment Anything Model: {sam_model_type}')
    
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # ** Optical flow model load
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

    # ** Depth estimation pipeline 로드
    print('Loading DepthAnythingV2 Model')
    depth_model = pipeline(task='depth-estimation',
                           model='depth-anything/Depth-Anything-V2-Small-hf', device=0)
    for idx, image_path in enumerate(tqdm.tqdm(image_files, desc="Processing Images")):
        print(f"\nProcessing image {idx+1}/{len(image_files)}: {image_path}")
        # first_image 
        first_image = Image.open(image_path) # depth image는 이걸 넣으면 됨
        
        # 1. Grounding DINO -> SAM 결과
        objects_info = grounding_dino_to_sam(dino_model=model, dino_processor=processor, sam_predictor=predictor, image=first_image, device=device)
        
        # 2. Hoist -> SAM 결과
        hoist_mask = hoist_bbox_sam(first_image, predictor)
        
        if hoist_mask is not None:
            print("Successfully generated mask from hoist-drawn bounding box.")
        else:
            print("Mask generation failed: Bounding box was not drawn or invalid.")
            # Distance 값을 None으로 설정하고 시각화
            distance_value = None
            output_image_path = os.path.join(args.output, f"distance_{os.path.basename(image_path)}")
            visualize_distance(first_image, distance_value, output_image_path)
            print(f"Saved distance visualization with N/A to: {output_image_path}")
            continue  # 다음 이미지로 넘어감
    
        print("SAM and Grounding DINO models successfully removed from memory.")
        
        # 3. Distance 계산 및 시각화
        distance_vals = distance(first_image, objects_info, hoist_mask, depth_model)  # 개별 거리 값 리스트 반환

        if distance_vals is not None:
            print(f"Calculated distances: {distance_vals}")
        else:
            print("Distance calculation failed.")

        # Distance 시각화
        output_image_path = os.path.join(args.output, f"distance_{os.path.basename(image_path)}")
        visualize_distance(first_image, distance_vals, output_image_path)
        print(f"Saved distance visualization to: {output_image_path}")

        
        # 4. Optical Flow (다음 이미지와 비교, 필요시)
        # 현재 모든 이미지에 대해 독립적으로 처리하고 있으므로, optical flow는 이전 이미지와 현재 이미지를 비교하는 식으로 수정
        if idx < len(image_files) - 1:
            second_image_path = image_files[idx + 1]
            second_image = Image.open(second_image_path)

            flag_b = optical_flow(first_image, second_image, objects_info, hoist_mask, device)
            print(f"Optical flow result between {image_path} and {second_image_path}: {flag_b}")

            if distance_value and flag_b:
                print("DANGEROUS")
        
        
        # SAM 모델 메모리 해제
        del predictor  # SAM Predictor 메모리에서 제거
        del sam  # SAM 모델 메모리에서 제거
        
        # Grounding DINO 모델 메모리 해제
        del model  # DINO 모델 메모리에서 제거
        del processor  # DINO Processor 메모리에서 제거
        
        # GPU 메모리 정리 (GPU에서 SAM 모델이 사용하던 메모리 해제)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("All images have been processed.")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Collision Prediction")
    parser.add_argument('--input', type=str, required=True,
                        help="Input image directory")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory to save results")
    args = parser.parse_args()

    main(args)
    

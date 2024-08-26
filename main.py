# https://github.com/Kyyle2114/SAM-Inference 참고

import os
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

    # 이미지 처리
    process_image(args.input, args.output, predictor, pipe,
                  args.mask_index, args.save_all_masks)

    print("Processing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM Inference Script")
    parser.add_argument('--input', type=str, required=True,
                        help="Input image path")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument('--model_type', type=str, default="vit_h", choices=[
                        "vit_h", "vit_l", "vit_b"], help="Model type for SAM (vit_h, vit_l, vit_b)")
    parser.add_argument('--mask_index', type=int, default=0,
                        choices=[0, 1, 2], help="Index for mask selection (0, 1, 2)")
    parser.add_argument('--save_all_masks', action='store_true',
                        help="Save results for all mask indices (0, 1, 2)")
    args = parser.parse_args()

    main(args)

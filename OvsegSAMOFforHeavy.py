import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "CAT-Seg"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "unimatch"))

import argparse
from glob import glob

import torch
import torch.nn.functional as F
from PIL import Image
# import matplotlib
# matplotlib.use('Agg')  # 파일 저장에 적합한 백엔드 설정
import matplotlib.pyplot as plt
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamPredictor, build_sam, SamAutomaticMaskGeneratorCustom # SAM

import time
import tqdm
import multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from cat_seg import add_cat_seg_config

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from types import SimpleNamespace as ns

from unimatch.unimatch.unimatch import UniMatch
from unimatch.utils.flow_viz import *

# constants
WINDOW_NAME = "MaskFormer demo"

class CATSegSegmentationMap(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, text=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        # set classes
        if text is not None:
            pred = self.predictor.model.sem_seg_head.predictor
            pred.test_class_texts = text.split(',')
            pred.text_features_test = pred.class_embeddings(pred.test_class_texts, 
                #imagenet_templates.IMAGENET_TEMPLATES, 
                 ['A photo of a {} in the scene',],
                pred.clip_model).permute(1, 0, 2).float().repeat(1, 80, 1)
            self.metadata = ns()
            self.metadata.stuff_classes = pred.test_class_texts
    
    def run_on_image_custom_text(self, image, text=None):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        
        if text is not None:
            pred = self.predictor.model.sem_seg_head.predictor
            print(text)
            pred.test_class_texts = text.split(',')
            pred.text_features_test = pred.class_embeddings(pred.test_class_texts, 
                #imagenet_templates.IMAGENET_TEMPLATES, 
                 ['A photo of a {} in the scene',],
                pred.clip_model).permute(1, 0, 2).float().repeat(1, 80, 1)
            self.metadata = ns()
            self.metadata.stuff_classes = pred.test_class_texts

        predictions = self.predictor(image)
        segmap = predictions['sem_seg'].argmax(dim=0)
        print(f"type(segmap) : {type(segmap)} \t shape: {segmap.shape}")
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device),
                    alpha=0.4,
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, segmap


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

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
    return parser

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

def load_unimatch_model(device, checkpoint_path):
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
        # task='flow'  # flow task 지정
    ).to(device)
    
    # model weight file path
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['model'], strict=False)
        print("Model weights loaded successfully.")
    
    flow_model.eval()
    return flow_model

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

    
def main(args):
    # --- args.input 검증 및 처리 (맨 앞에 위치) ---
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

    # --- CAT-Seg 및 SAM 로딩 ---
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    catseg_map = CATSegSegmentationMap(cfg)
    
    text = 'floor, person, forklift, machine, wall, ceiling' 
    target_class = 0 # floor's index: 0
    
    # SAM load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_checkpoint = "sam_vit_h_4b8939.pth" # "sam_vit_l_0b3195.pth", "sam_vit_b_01ec64.pth"
    sam_model_type = "vit_h"

    print(f'Loading Segment Anything Model: {sam_model_type}')
    
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # --- UniMatch Optical Flow 모델 로드 ---
    flow_model_checkpoint = "unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"  # Optical Flow 모델 경로
    flow_model = load_unimatch_model(device, flow_model_checkpoint)
    
    # --- Optical Flow 인자 설정 ---
    flow_args = argparse.Namespace(
        padding_factor=32,         # 패딩 팩터 설정
        inference_size=None,       # 추론 크기
        pred_bwd_flow=False,
        attn_type='swin',
        attn_splits_list=[2, 8],   # attention split list
        corr_radius_list=[-1, 4],  # correlation 반경 리스트
        prop_radius_list=[-1, 1],  # propagation 반경 리스트
        num_reg_refine=6,           # refinement 스텝 개수
        task='flow',
        device=device
    )
    
    # --- 이미지 처리 루프 ---
    for idx, path in enumerate(tqdm.tqdm(input_paths, desc="Processing Images")):
        # try:
            # 이미지 로드
            img = read_image(path, format="BGR")
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_copy = image.copy()

            # 다음 이미지 로드 (Optical Flow를 위해)
            if idx < len(input_paths) - 1:
                next_path = input_paths[idx + 1]
                next_img = read_image(next_path, format="BGR")
                next_image = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)
            else:
                next_image = None
            
            start_time = time.time()
            predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)
            
            mask_generator_custom = SamAutomaticMaskGeneratorCustom(sam, semantic_map=segmap, target_class=target_class)
            masks = mask_generator_custom.generate(image)
            print(f"Mask keys: {masks[0].keys()}")  

            # --- Drawing Bbox & y_threshold ---
            drawing = False  # 마우스 버튼이 눌렸는지 여부
            start_point = (-1, -1)  # BBox의 시작점
            end_point = (-1, -1)  # BBox의 끝점

            # 마우스 콜백 함수 정의
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
                x_threshold_min = min(x1, x2)
                x_threshold_max = max(x1, x2)
                print(f"Selected BBox: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"x_threshold_min: {x_threshold_min}, x_threshold_max: {x_threshold_max}")
            else:
                print("BBox가 선택되지 않았습니다. 모든 마스크를 유지합니다.")
                x_threshold_min = 0
                x_threshold_max = image.shape[1]

            # --- 마스크 필터링 ---
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

            filtered_masks = []
            for mask in masks:
                if mask_overlaps_bbox_x(mask['segmentation'], x_threshold_min, x_threshold_max):
                    filtered_masks.append(mask)

            print(f"전체 마스크 수: {len(masks)}, 필터링된 마스크 수: {len(filtered_masks)}")

            # --- Optical Flow 계산 ---
            if next_image is not None:
                # 이미지 Tensor로 변환
                image1_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image2_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # Optical Flow 계산
                flow_pr, ori_size = optical_flow_uni_match(flow_model, image1_tensor.to(device), image2_tensor.to(device), flow_args)
                
                print(f"type(flow_pr): {type(flow_pr)}, shape: {flow_pr.shape}")
                # Optical Flow 결과의 Magnitude(크기) 계산
                flow_magnitude = np.linalg.norm(flow_pr, axis=2)
                
                # flow_magnitude를 ori_size에 맞게 리사이즈
                flow_magnitude = cv2.resize(flow_magnitude, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_LINEAR)
                print(f"flow_magnitude_resized shape: {flow_magnitude.shape}")
            else:
                print("다음 이미지가 없어 Optical Flow를 계산할 수 없습니다.")
                flow_magnitude = None

            # --- Optical Flow 기반 추가 마스크 필터링 ---
            if flow_magnitude is not None:
                def mask_has_flow(mask, flow_mag, threshold=1.0):
                    """
                    마스크 영역 내에 Optical Flow가 존재하는지 확인합니다.

                    Args:
                        mask (np.ndarray): 2D 이진 마스크 배열.
                        flow_mag (np.ndarray): Optical Flow Magnitude 배열.
                        threshold (float): Flow magnitude 임계값.

                    Returns:
                        bool: 마스크 영역 내에 임계값을 초과하는 flow가 있으면 True, 아니면 False.
                    """
                    # 마스크 영역 내의 Optical Flow Magnitude 추출
                    flow_in_mask = flow_mag[mask > 0]
                    if flow_in_mask.size == 0:
                        return False
                    # 임계값을 초과하는 흐름 벡터가 있는지 확인
                    return np.any(flow_in_mask > threshold)

                # 추가 필터링된 마스크 리스트 생성
                final_filtered_masks = []
                for mask in filtered_masks:
                    if mask_has_flow(mask['segmentation'], flow_magnitude, threshold=0.0):  # threshold 값은 필요에 따라 조정
                        final_filtered_masks.append(mask)

                print(f"Optical Flow를 고려한 최종 마스크 수: {len(final_filtered_masks)}")
            else:
                # Optical Flow가 없으면 모든 마스크를 제거
                final_filtered_masks = []
                print("Optical Flow가 존재하지 않아 모든 마스크를 제거합니다.")
            
            # --- BBox 그리기 ---
            image_with_bboxes = image.copy()
            cv_image_with_bboxes = cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv_image_with_bboxes, start_point, end_point, (0, 255, 0), 2)
            image_with_bboxes = cv2.cvtColor(cv_image_with_bboxes, cv2.COLOR_BGR2RGB)

            # --- 선택된 마스크 시각화 ---
            plt.figure(figsize=(10,10))
            plt.imshow(image_with_bboxes)
            # final_filtered_masks를 show_anns에 전달하여 마스크 시각화
            show_anns(final_filtered_masks, alpha=0.5)
            plt.axis('off')
            plt.title("Filtered Masks Below BBox with Optical Flow Consideration")
            
            
            # --- 시각화된 이미지 저장 또는 표시 ---
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                
                # 출력 파일 경로 설정
                if os.path.isdir(args.output):
                    # 출력 디렉토리가 존재하는 경우, 동일한 파일 이름으로 저장
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    # 출력이 디렉토리가 아닌 경우, 단일 파일로 저장 (단일 이미지 처리 시 유용)
                    out_filename = args.output
                plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
                plt.close()  # 메모리 관리 위해 그림 닫기
                print(f"Saved visualization to: {out_filename}")
            else:
                plt.show()
                plt.close()

            # --- 로깅 ---
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            print(f"\npredictions shape: {predictions['sem_seg'].shape}\n")

            # --- Optical Flow 처리 (다음 이미지와 비교, 필요시) ---
            # Optical Flow 관련 코드가 필요하다면 여기에 추가
        # except Exception as e:
        #     print(f"Error processing {path}: {e}")
        #     continue
        
    print("All images have been processed.")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
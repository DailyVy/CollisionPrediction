import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "CAT-Seg"))

import argparse
from glob import glob

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamPredictor, build_sam, SamAutomaticMaskGeneratorCustom # SAM
# from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorCustom
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.plotting import Annotator, colors


import tempfile
import time
import warnings

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

def filter_masks_by_segmap(masks, segmap, target_class=0):
    """
    Filters FastSAM masks based on overlap with target class in segmap
    Args:
        masks: FastSAM mask results
        segmap: Semantic segmentation map from CAT-Seg
        target_class: Target class index (0 for floor)
    Returns:
        filtered_masks: List of masks that overlap with target class
    """
    filtered_masks = []
    target_area = (segmap == target_class).cpu().numpy()
    
    for mask in masks:
        mask_np = mask.data.cpu().numpy()
        # Calculate overlap with target class
        overlap = np.logical_and(mask_np.squeeze(), target_area)
        if np.any(overlap):
            filtered_masks.append(mask_np)
    
    return filtered_masks

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
    
def show_masks(masks, alpha=0.35):
    """
    Visualizes FastSAM masks
    Args:
        masks: FastSAM mask results (Tensor shape [N, H, W])
        alpha: Transparency value for visualization
    """
    if len(masks) == 0:
        return
        
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    # Create empty RGBA image
    h, w = masks[0].shape[1:]  # Get height and width from first mask
    img = np.ones((h, w, 4))
    img[:,:,3] = 0
    
    # Convert each mask tensor to numpy and visualize
    for mask in masks:
        # Convert tensor mask to numpy boolean array
        m = mask.squeeze().astype(bool)
        
        # Generate random color with alpha
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
        
    ax.imshow(img)

if __name__ == "__main__":
    # OV-Seg: CAT-Seg
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    catseg_map = CATSegSegmentationMap(cfg)
    
    text = 'floor, person, forklift, machine, wall, ceilling' 
    target_class = 0 # floor's index: 0
    
    # Segmentation: SAM load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_sam = FastSAM("FastSAM-s.pt")
    print(f'Loading Fast Segment Anything Model: FastSAM-s.pt')

    if args.input:
        input_paths = []
        print(args.input)
        if os.path.isdir(args.input[0]):
            # 디렉토리인 경우, 디렉토리 내의 모든 이미지 파일을 리스트로 가져옴
            input_paths = sorted(
                glob(os.path.join(os.path.expanduser(args.input[0]), "*.png")) + 
                glob(os.path.join(os.path.expanduser(args.input[0]), "*.jpg")) +
                glob(os.path.join(os.path.expanduser(args.input[0]), "*.jpeg"))
            )
            assert input_paths, f"No image files found in directory: {args.input[0]}"
        elif os.path.isfile(args.input[0]):
            input_paths = args.input
        else:
            raise ValueError(f"Input path is neither a directory nor a file: {args.input[0]}")

        for path in tqdm.tqdm(input_paths, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_copy = image.copy()
            
            start_time = time.time()
            predictions, visualized_output, segmap = catseg_map.run_on_image_custom_text(img, text)
            
            everything_results = fast_sam(image, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)[0]
            masks = everything_results.masks

            # segmap 기반 필터링 적용
            masks = filter_masks_by_segmap(masks, segmap, target_class=0)
            print(f"mask: {type(masks)}") # <class 'ultralytics.engine.results.Masks'>
            print(f"mask: {masks[0].shape}") #  torch.Size([1, 360, 640])

            # --- Drawing Bbox & y_threshold ---
            drawing = False  # 마우스 버튼이 눌렸는지 여부
            start_point = (-1, -1)  # BBox의 시작점
            end_point = (-1, -1)  # BBox의 끝점
            # image_copy = img.copy()

            # 마우스 콜백 함수 정의
            def draw_rectangle(event, x, y, flags, param):
                global start_point, end_point, drawing, image_copy

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
                # cv2.imshow('Draw BBox', cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
                cv2.imshow('Draw BBox', cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
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
                x_threshold_max = img.shape[1]

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
                if mask_overlaps_bbox_x(mask.squeeze(0), x_threshold_min, x_threshold_max):
                    filtered_masks.append(mask)
            
            # for mask_data in masks:
            #     mask = mask_data.data.cpu().numpy()
            #     print("mask type:", type(mask)) # mask type: <class 'numpy.ndarray'>
            #     print("mask shape:", np.shape(mask)) # mask shape: (1, 360, 640)
            #     if mask_overlaps_bbox_x(mask.squeeze(0), x_threshold_min, x_threshold_max):
            #         filtered_masks.append(mask)

            print(f"전체 마스크 수: {len(masks)}, 필터링된 마스크 수: {len(filtered_masks)}")

            # --- BBox 그리기 ---
            image_with_bboxes = image.copy()
            cv_image_with_bboxes = cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv_image_with_bboxes, start_point, end_point, (0, 255, 0), 2)
            image_with_bboxes = cv2.cvtColor(cv_image_with_bboxes, cv2.COLOR_BGR2RGB)

            # --- 선택된 마스크 시각화 ---
            plt.figure(figsize=(10,10))
            plt.imshow(image_with_bboxes)
            show_masks(filtered_masks, alpha=0.5)
            plt.axis('off')
            plt.title("Filtered Masks Below BBox")
            
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
            
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                # visualized_output.save(out_filename)
                plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
                plt.show()
                plt.close()
            else:
                plt.show()
                plt.close()
            



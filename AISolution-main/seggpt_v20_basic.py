import os
import argparse

import cv2
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms


import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
from typing import List, Dict, Tuple
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import threading

from matplotlib.path import Path
import matplotlib
import time

######## 외부 함수 ###########3
import models_seggpt

from depth_anything_v2.dpt import DepthAnythingV2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator,colors

from tracker import*
from hikvisionPTZ import go_to_position, get_position

###########################################################################
# SegGPT 용 함수 
###########################################################################
class Cache(list):
    def __init__(self, max_size=0):
        super().__init__()
        self.max_size = max_size

    def append(self, x):
        if self.max_size <= 0:
            return
        super().append(x)
        if len(self) > self.max_size:
            self.pop(0)


@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)


    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # nomalize
    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def inference_image(model, device, img, img2_paths, tgt2_paths):
    center = None
    res, hres = 448, 448

    image = Image.fromarray(img)
    input_image = image
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)

    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)

    # tensor -> image
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()
    
    # 마스크를 바이너리로 변환
    threshold = 128  # 임계값 설정
    if output.ndim == 3 and output.shape[-1] == 3:  # 만약 3채널 이미지라면
      output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2GRAY)  # 단일 채널(그레이스케일)로 변환

    _, binary_mask = cv2.threshold(output, threshold, 255, cv2.THRESH_BINARY)
    # 3. 마스크 영역 감지
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 각 마스크의 중앙 좌표 계산 및 그리기
    input_image_cv = np.array(input_image.convert('RGB'))  # PIL 이미지를 numpy 배열로 변환

    coords = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # 면적이 0이 아닌 경우
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            center = (cX, cY)
            coords.append(center)
            # 5. 포인트 그리기 (중앙 좌표에 빨간 점 표시)
            cv2.circle(input_image_cv, center, radius=10, color=(0, 0, 255), thickness=-1)

    return input_image_cv,coords

def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


###########################################################################
# Depth 함수
###########################################################################
def depth_inference_image(frame) :
    depth = depth_model.infer_image(frame)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return depth


###########################################################################
# 보조 함수 
###########################################################################
def draw_detections(image, result, box_conf = False, show_conf = False, mask_conf = True, line_width = 1, font_size = 1):
        annotator = Annotator(image, line_width, font_size, 'Arial.ttf', False, 'example')
        boxes = result.boxes
        masks = result.masks
        probs = result.probs
        names = result.names
        hide_labels, hide_conf = False, not show_conf

        masks_resized = None
        points = []
        if boxes is not None :
            for d in reversed(boxes):
                x1, y1, x2, y2 = map(int, d.xyxy[0])
                center = (((x1 + x2) / 2), ((y1 + y2) / 2))
                points.append(center)

                if box_conf is True :
                    c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                    name = ('' if id is None else f'id:{id} ') + names[c]
                    label = None if hide_labels else (name if hide_conf else f'{name} {conf:.2f}')
                    annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        if masks is not None and mask_conf:
            # 이미지와 마스크의 크기를 가져옵니다.
            image_shape = annotator.im.shape[:2]  # 이미지의 높이와 너비 가져오기
            mask_shape = masks.data.shape[1:]  # 마스크의 높이와 너비 가져오기

            # 마스크를 이미지 크기로 리사이즈
            masks_resized = F.interpolate(
                masks.data[None, ...], 
                size=[image_shape[0], image_shape[1]], 
                mode='nearest'
            ).permute(0, 1, 2, 3)[0]
            # 마스크를 이미지에 적용
            if masks_resized is not None:
                im = torch.as_tensor(annotator.im, dtype=torch.float16, device=masks.data.device).permute(2, 0, 1).flip(0)
                im = im / 255  # 이미지를 [0, 1] 범위로 정규화
                annotator.masks(masks_resized, colors=[colors(x, True) for x in boxes.cls], im_gpu=im)
                        
        return annotator.im, masks_resized, points
def compare_mask() :
    pass 

def draw_detectionLine(img, coords):
    color_ = [(255,0,0), (0,0,255), (0,255,0), (255,0,255), (255,255,0)]
    index = 0

    coords = convert_ratio2xy(img,coords)
    # numpy 배열로 변환
    pts = np.array(coords, dtype=np.int32)

    # check_area_coords 함수가 좌표가 올바른지 확인
    if check_area_coords(coords):
        # cv2.polylines(원본그림, 좌표리스트, 마지막점과 첫점 연결 여부, 선색, 선 굵기)
        cv2.polylines(img, [pts], True, color_[index % len(color_)], 2)

    index += 1

    return img

def calculate_camera_index(url) :
    import math
    base = url.find('Channels/')
    camera_index = math.floor(int(url[base+9:]) / 100)
    return camera_index - 1
###########################################################################
# 조건 탐색 함수
###########################################################################
def check_area_coords(coords) :
    isValid = False
    is_exist = 0
    for coord in coords :
        if coord[0] != -1 and coord[1] != -1 :
            is_exist = is_exist + 1
    
    isValid = True if is_exist == 4 else False

    return isValid 

def convert_ratio2xy(img, coords) :
    # 이미지 크기 가져오기
    img_height, img_width = img.shape[:2]
    # 비율 좌표를 실제 좌표로 변환
    abs_coords = []
    for coord in coords:
        # 비율 좌표를 실제 픽셀 좌표로 변환
        x = int(coord[0] * img_width)
        y = int(coord[1] * img_height)
        abs_coords.append((x, y))
    return abs_coords
#######################################################
# 범위내 진입, 호이스트 가동범위 이탈
#######################################################
def detection_hoistDeparture(img,mask,coords,detect_count) :
    alert = False
    if check_area_coords(coords) and mask is not None:
        image_shape = img.shape[:2]  # 이미지의 높이와 너비 가져오기

        coords = convert_ratio2xy(img,coords)

        coord_mask = create_mask_from_coords(coords, image_shape)
        iou = calculate_iou(coord_mask, mask)
        if iou > 0:
            alert = False
        else : 
            detect_count = detect_count + 1
            if detect_count > 3 : 
                alert = True
                detect_count = 0

    return img, alert

def detection_dangerZone(img,mask,coords) :
    alert = False
    if check_area_coords(coords) and mask is not None:
        image_shape = img.shape[:2]  # 이미지의 높이와 너비 가져오기

        coords = convert_ratio2xy(img,coords)

        coord_mask = create_mask_from_coords(coords, image_shape)
        iou = calculate_iou(coord_mask, mask)
        if iou > 0:
            # 특정 조건을 충족하는 객체 처리
            alert = True
        else : 
            alert = False
    return img, alert

def calculate_iou(mask1, mask2):
    # mask1이 텐서라면, CPU로 옮기고 NumPy 배열로 변환
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    
    # mask2가 텐서라면, CPU로 옮기고 NumPy 배열로 변환
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # IoU 계산
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def create_mask_from_coords(coords, image_shape):
    # 좌표 배열을 numpy 배열로 변환하고, 부동소수점을 정수로 변환
    coords = np.array(coords, dtype=np.float32)
    coords = coords.astype(np.int32)
    # 디버깅용: 좌표 배열의 형태를 출력
    #print(f"Coordinates shape: {coords.shape}, dtype: {coords.dtype}")
    # 좌표 배열의 형식이 올바른지 확인
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Invalid coordinates format. Got {coords.shape}, expected (N, 2).")
    # 마스크 초기화
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # 폴리곤 그리기
    cv2.fillPoly(mask, [coords], 1)

    return mask

#######################################################
# 입출입
#######################################################
def detection_entryExit(img, result, coords) :
    alert = False
    global person_in_count
    global person_out_count
    offset=6
    list=[]

    boxes = result.boxes
    names = result.names

    coords = convert_ratio2xy(img,coords)

    line1_start = (int((2 * coords[0][0] + coords[1][0])/3), int((2 * coords[0][1] + coords[1][1])/3))
    line1_end   = (int((coords[2][0] + 2 * coords[3][0])/3), int((coords[2][1] + 2 * coords[3][1])/3))
    line2_start = (int((coords[0][0] + 2 * coords[1][0])/3), int((coords[0][1] + 2 * coords[1][1])/3))
    line2_end   = (int((2 * coords[2][0] + coords[3][0])/3), int((2 * coords[2][1] + coords[3][1])/3))

    cx1 = ( line1_start[0] + line1_end[0] ) / 2
    cx2 = ( line2_start[0] + line2_end[0] ) / 2

    cv2.line(img, line1_start, line1_end, (0,255,0),    2)
    cv2.line(img, line2_start, line2_end, (255,255,0),  2)

    if boxes is not None :
        for d in reversed(boxes):
            c, id = int(d.cls), None if d.id is None else int(d.id.item())
            name = ('' if id is None else f'id:{id} ') + names[c]
            xyxy = d.xyxy.cpu().detach().numpy().tolist()[0]
            x1=xyxy[0]
            y1=xyxy[1]
            x2=xyxy[2]
            y2=xyxy[3]
            if 'person' in name:
                list.append([x1,y1,x2,y2])
        bbox_id=tracker.update(list)
        for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2

            if cx1 < ( cx + offset) and cx1 > (cx - offset) :
                person_in_buf[id] = cx
            if id in person_in_buf :
                if cx2 < ( cx + offset) and cx2 > (cx - offset) :
                    cv2.circle(img,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    if person_in_list.count(id) == 0 :
                        if len(person_in_list) > 100 :
                            person_in_list.clear()
                        person_in_list.append(id)
                        person_in_count = person_in_count + 1
                        alert = True

            if cx2 < ( cx + offset) and cx2 > (cx - offset) :
                person_out_buf[id] = cx
            if id in person_out_buf :
                if cx1 < ( cx + offset) and cx1 > (cx - offset) :
                    cv2.circle(img,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    if person_out_list.count(id) == 0 :
                        if len(person_out_list) > 100 :
                            person_out_list.clear()
                        person_out_list.append(id)
                        person_out_count = person_out_count + 1
                        alert = True
    return img, person_in_count, person_out_count, alert

###########################################################################
# 영상처리 수행 함수 
###########################################################################
def process_video(queue: Queue, url: str):
    while True:
        try:
            frame = queue.get(timeout=1)  # Wait for a frame리
        except Empty:
            continue

        # Camera index calculation
        camera_index = calculate_camera_index(url)
        
        # Resize (change to desired size)
        resize_width, resize_height = 1280, 720
        frame = cv2.resize(frame, (resize_width, resize_height))

        original = frame.copy()
        img2_path = ['examples/sun7.jpg']
        tgt2_path = ['examples/sun7_target.png']

        relevant_objects = []
        alert = False

        crossed_in_counter = 0
        crossed_out_counter = 0
        hook_xy = None
        yolo_xy = None

        # # Object detection and tracking (replace with your own model inference code)
        img_result,hook_xy = inference_image(fewshot_model, DEVICE, original, img2_path, tgt2_path)
        yolo_results = yolo_model.track(original, classes=0, conf=0.2, iou=0.5, device=DEVICE, persist=True, tracker="bytetrack.yaml",verbose=False)[0]
        depth_results = depth_inference_image(original)
        
        img_result, mask, yolo_xy = draw_detections(img_result, yolo_results, box_conf=False, mask_conf=True)
        img_depth, _, _ = draw_detections(depth_results, yolo_results)


        ai_setting = ai_settings[calculate_camera_index(url)]
        # Optical Flow (if enabled)
        if ai_setting.hoistOperationDetection:
            user_points = []
            user_point = np.empty([0, 1, 2], dtype=np.float32)

            # if hook_xy is not None:
            #     for xy in hook_xy : 
            #         user_point = np.empty([1, 1, 2], dtype=np.float32)
            #         user_point[0][0] = [hook_xy[0], hook_xy[1]]
            #         user_points.append(user_point)

            # if yolo_xy is not None:
            #     for xy in yolo_xy:
            #         user_point = np.empty([1, 1, 2], dtype=np.float32)
            #         user_point[0][0] = [xy[0], xy[1]]
            #         user_points.append(user_point)

            img_result = detection_movement(img_result, user_points)

        # # Entry/Exit Detection (if enabled)
        # if ai_setting.entryExitDetection.enabled:
        #     draw_detectionLine(frame, ai_setting.entryExitDetection.area)
        #     frame, crossed_in_counter, crossed_out_counter, alert = detection_entryExit(frame, yolo_results, ai_setting.entryExitDetection.area)

        # # Danger Zone Detection (if enabled)
        # if ai_setting.dangerZoneDetection.enabled:
        #     draw_detectionLine(frame, ai_setting.dangerZoneDetection.area)
        #     frame, alert = detection_dangerZone(frame, mask, ai_setting.dangerZoneDetection.area)

        # # Hoist Departure Detection (if enabled)
        # if ai_setting.hoistDepartureDetection.enabled:
        #     draw_detectionLine(frame, ai_setting.hoistDepartureDetection.area)
        #     frame, alert = detection_hoistDeparture(frame, mask, ai_setting.hoistDepartureDetection.area, ai_setting.hoistDepartureDetection.count)

        # Convert frame to base64
        _, buffer_ori = cv2.imencode('.jpg', original)
        ori_str = base64.b64encode(buffer_ori).decode('utf-8')

        _, buffer_res = cv2.imencode('.jpg', img_result)
        img_str = base64.b64encode(buffer_res).decode('utf-8')

        _, buffer_depth = cv2.imencode('.jpg', img_depth)
        depth_str = base64.b64encode(buffer_depth).decode('utf-8')

        yield {
            "camera_index": camera_index,
            "original_image": ori_str,
            "result_image": img_str,
            "depth_image": depth_str,
            "object_count": f"In: {crossed_in_counter} / Out: {crossed_out_counter}",
            "relevant_objects": relevant_objects,
            "alert": alert
        }


app = FastAPI()

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

fewshot_model = prepare_model('checkpoints/seggpt_vit_large.pth', 'seggpt_vit_large_patch16_input896x448', 'semantic').to(DEVICE)
yolo_model = YOLO("yolov8x-seg.pt")

# Store the coordinates for each URL
coordinates: List[Tuple[int, int]] = {}
scales : List = {}

# Depth model config
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_model = DepthAnythingV2(**model_configs['vitl'])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

tracker=Tracker()
person_out_count = 0
person_in_count = 0
person_out_buf = {}
person_out_list = []
person_in_buf = {}
person_in_list = []
###########################################################################
# Optical Flow
###########################################################################

optical_color = np.random.randint(0,255,(250,3))
optical_lines = None  #추적 선을 그릴 이미지 저장 변수
optical_prevImg = None  # 이전 프레임 저장 변수
optical_prevPt = None

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.1,
                       minDistance = 10,
                       blockSize = 7,
                       useHarrisDetector = True,
                       k = 0.03
                       )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



def detection_movement(frame, user_points) :
    global optical_color
    global optical_lines
    global optical_prevImg
    global optical_prevPt
    
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우
    if optical_prevImg is None:
        optical_prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        optical_lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출  ---①
        optical_prevPt = cv2.goodFeaturesToTrack(optical_prevImg, mask=None, **feature_params)


        if len(user_points) > 0:
            for user_point in user_points :
                optical_prevPt = np.concatenate([optical_prevPt, user_point])
                user_point = np.empty([0,1,2])
    else:
        nextImg = gray

        # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(optical_prevImg, nextImg, optical_prevPt, None, **lk_params)
        if nextPt is not None : 
            # 대응점이 있는 코너, 움직인 코너 선별 ---③
            prevMv = optical_prevPt[status==1]
            nextMv = nextPt[status==1]
            for i,(p, n) in enumerate(zip(prevMv, nextMv)):
                px,py = p.ravel()
                nx,ny = n.ravel()

                px = int(px)
                py = int(py)
                nx = int(nx)
                ny = int(ny)
                # 이전 코너와 새로운 코너에 선그리기 ---④
                cv2.line(optical_lines, (px, py), (nx,ny), optical_color[i].tolist(), 2)
                # 새로운 코너에 점 그리기
                cv2.circle(img_draw, (nx,ny), 2, optical_color[i].tolist(), -1)

            # 누적된 추적 선을 출력 이미지에 합성 ---⑤
            img_draw = cv2.add(img_draw, optical_lines)
            # 다음 프레임을 위한 프레임과 코너점 이월
            optical_prevImg = nextImg
            optical_prevPt = nextMv.reshape(-1,1,2)

    return img_draw
###########################################################################
# 설정 변수
###########################################################################
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class CollisionDetection:
    enabled: bool = False
    sensitivity: str = "medium"

@dataclass
class AreaDetection:
    enabled: bool = False
    area: List[List[int]] = field(default_factory=lambda: [[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
    count : int = 0

@dataclass
class CameraSettings:
    cameraUrl : str = ""
    humanDetection: bool = True
    safetyGearDetection: bool = False
    hoistOperationDetection: bool = False
    forkliftCollisionDetection: CollisionDetection = field(default_factory=CollisionDetection)
    heavyObjectCollisionDetection: CollisionDetection = field(default_factory=CollisionDetection)
    entryExitDetection: AreaDetection = field(default_factory=AreaDetection)
    equipmentCollisionDetection: AreaDetection = field(default_factory=AreaDetection)
    hoistDepartureDetection: AreaDetection = field(default_factory=AreaDetection)
    dangerZoneDetection: AreaDetection = field(default_factory=AreaDetection)
    pedestrianPassageDetection : AreaDetection = field(default_factory=AreaDetection)

# Example of how to create settings for a camera
ai_settings : List[CameraSettings] = []

class PTZCommand(BaseModel):
    url: str
    channel : int
    id: str
    password: str
    command: str

# Global variable to store current PTZ position
@dataclass
class PTZSettings: 
    initialze : bool = False
    url : str = ""
    id : str = ""
    password : str = ""
    pan : int = 0
    tilt : int =  0
    zoom : int = 0

ptz_settings = PTZSettings()

def update_ai_settings(detection_settings) : 
    global ai_settings
    global optical_prevImg
    optical_prevImg = None

    if len(ai_settings) == 0 :
        for id in range(12) : 
            ai_setting = CameraSettings()
            ai_settings.append(ai_setting)

    for id in detection_settings : 
        index = calculate_camera_index(id)

        detection_setting = detection_settings.get(id,True)

        ai_settings[index].cameraUrl = detection_setting.get('cameraUrl',True)
        ai_settings[index].humanDetection = detection_setting.get('humanDetection',True)
        ai_settings[index].safetyGearDetection = detection_setting.get('safetyGearDetection',True)
        ai_settings[index].hoistOperationDetection = detection_setting.get('hoistOperationDetection',True)
        
        setting = detection_setting.get('forkliftCollisionDetection', True)
        ai_settings[index].forkliftCollisionDetection.enabled = setting.get('enabled',True)
        ai_settings[index].forkliftCollisionDetection.sensitivity = setting.get('sensitivity',True)


        setting = detection_setting.get('heavyObjectCollisionDetection', True)
        ai_settings[index].heavyObjectCollisionDetection.enabled = setting.get('enabled',True)
        ai_settings[index].heavyObjectCollisionDetection.sensitivity = setting.get('sensitivity',True)

        setting = detection_setting.get('entryExitDetection', True)
        ai_settings[index].entryExitDetection.enabled = setting.get('enabled',True)
        ai_settings[index].entryExitDetection.area = setting.get('area',True)

        setting = detection_setting.get('equipmentCollisionDetection', True)
        ai_settings[index].equipmentCollisionDetection.enabled = setting.get('enabled',True)
        ai_settings[index].equipmentCollisionDetection.area = setting.get('area',True)

        setting = detection_setting.get('hoistDepartureDetection', True)
        ai_settings[index].hoistDepartureDetection.enabled = setting.get('enabled',True)
        ai_settings[index].hoistDepartureDetection.area = setting.get('area',True)

        setting = detection_setting.get('dangerZoneDetection', True)
        ai_settings[index].dangerZoneDetection.enabled = setting.get('enabled',True)
        ai_settings[index].dangerZoneDetection.area = setting.get('area',True)

        setting = detection_setting.get('pedestrianPassageDetection', True)
        ai_settings[index].pedestrianPassageDetection.enabled = setting.get('enabled',True)
        ai_settings[index].pedestrianPassageDetection.area = setting.get('area',True)

###########################################################################
# FastAPI 용 함수 
###########################################################################

@app.post("/ptz")
async def process_ptz_command(command: PTZCommand):
    pan_increment = 5  # Example values, adjust as needed
    tilt_increment = 5
    zoom_increment = 1

    if not ptz_settings.initialze :
        ptz_settings.initialze = True
        ptz_settings.pan, ptz_settings.tilt, ptz_settings.zoom = get_position(command.url, command.channel, command.id, command.password)

    if command.command == "pan_left":
        ptz_settings.pan -= pan_increment
    elif command.command == "pan_right":
        ptz_settings.pan += pan_increment
    elif command.command == "tilt_up":
        ptz_settings.tilt += tilt_increment
    elif command.command == "tilt_down":
        ptz_settings.tilt -= tilt_increment
    elif command.command == "zoom_in":
        ptz_settings.zoom += zoom_increment
    elif command.command == "zoom_out":
        ptz_settings.zoom -= zoom_increment

    # Ensure the values stay within bounds (0-360 for pan/tilt, zoom level depending on camera)
    ptz_settings.pan = max(0, min(3550, ptz_settings.pan))
    ptz_settings.tilt = max(0, min(900, ptz_settings.tilt))
    ptz_settings.zoom = max(0, min(640, ptz_settings.zoom))

    # Apply the new PTZ position
    go_to_position(command.url, command.channel, command.id, command.password, ptz_settings.pan, ptz_settings.tilt, ptz_settings.zoom)


@app.get("/optical")
async def process_optical_command():
    global optical_prevImg
    optical_prevImg = None

############################################################################
class VideoCaptureThread(threading.Thread):
    def __init__(self, url, queue):
        super().__init__()
        self.url = url
        self.queue = queue
        self.stop_event = threading.Event()

    def run(self):
        try:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.stop_event.set()
                raise ValueError(f"Cannot open RTSP stream: {self.url}")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()  # Discard previous frame
                    except Empty:
                        pass
                self.queue.put(frame)
        except Exception as e:
            print(f"Error in video_capture: {e}")
        finally:
            cap.release()

    def stop(self):
        self.stop_event.set()

def stop_threads(threads):
    """Stop all threads cleanly."""
    for thread in threads:
        if thread.is_alive():
            thread.stop()  # Signal the thread to stop
            thread.join()  # Wait for the thread to actually stop

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        #self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        await websocket.close()
        #self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, websocket : WebSocket, message: str, timeout: int = 5):
        try:
            await asyncio.wait_for(websocket.send_text(message), timeout)
        except asyncio.TimeoutError:
            print(f"Sending message to a connection timed out after {timeout} seconds")

manager = ConnectionManager()

async def websocket_receiver(websocket: WebSocket, manager, queues, urls, sender_task_ref, threads):
    """Continuously receive messages from the websocket."""
    try:
        while True:
            data = await websocket.receive_text()  # Receive a message
            request = json.loads(data)

            detection_settings = request["detectionSettings"]
            if detection_settings:
                print(detection_settings)
                update_ai_settings(detection_settings)

                # Ensure sender task is running and reset it if necessary
                if sender_task_ref[0]:
                    sender_task_ref[0].cancel()  # Cancel the existing sender task
                    try:
                        await sender_task_ref[0]  # Ensure the task is cancelled properly
                    except asyncio.CancelledError:
                        print("Sender task cancelled and will be restarted.")

                    # Restart the sender task with updated settings
                    sender_task_ref[0] = asyncio.create_task(websocket_sender(websocket, manager, queues, urls))

            new_urls = request.get("urls")
            # If new URLs are provided, update queues and start new threads
            if new_urls:
                # Stop existing threads and clear queues
                stop_threads(threads)
                queues.clear()
                urls.clear()

                # Update URLs
                for url in new_urls:
                    urls.append(url)
                    queue = Queue(maxsize=10)
                    queues[url] = queue
                    t = VideoCaptureThread(url, queue)
                    t.start()
                    threads.append(t)
                    print(f"Started video capture thread for {url}")

                # Cancel the existing sender task and restart it with the updated URL list
                if sender_task_ref[0]:
                    sender_task_ref[0].cancel()  # Cancel the existing sender task
                    try:
                        await sender_task_ref[0]  # Ensure the task is cancelled properly
                    except asyncio.CancelledError:
                        print("Sender task cancelled and will be restarted.")

                # Restart the sender task with updated URLs
                sender_task_ref[0] = asyncio.create_task(websocket_sender(websocket, manager, queues, urls))

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        print("WebSocket disconnected during receiving.")
    except Exception as e:
        print(f"Error in receiver: {e}")

async def websocket_sender(websocket: WebSocket, manager, queues, urls):
    """Handle sending messages to the websocket."""
    tasks = []
    try:
        if len(urls) > 0:  # Ensure there are URLs before creating ThreadPoolExecutor
            # Start processing videos using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(urls)) as executor:
                futures = {executor.submit(process_video, queue, url): url for url, queue in queues.items()}
                while True:
                    for future in futures:
                        url = futures[future]
                        try:
                            result = next(future.result(), None)
                            if result:
                                result["url"] = url  # Add URL to the result

                                # Create a task for broadcasting with a timeout
                                broadcast_task = asyncio.create_task(manager.broadcast(websocket, json.dumps(result), 3))
                                tasks.append(broadcast_task)
                                await broadcast_task
                        except StopIteration:
                            continue

        else:
            print("No URLs available for processing.")
            await asyncio.sleep(1)  # Sleep briefly before checking again

    except WebSocketDisconnect:
        print("WebSocket disconnected during sending.")
    except RuntimeError as e:
        # Handle RuntimeError
        if str(e) == 'Cannot call "send" once a close message has been sent.':
            print("Attempted to send a message after WebSocket was closed.")
        else:
            raise e
    finally:
        # Ensure that all tasks are cancelled if not already
        for task in tasks:
            if not task.done():
                task.cancel()

async def websocket_handler(websocket: WebSocket):
    await manager.connect(websocket)
    urls = []  # Initialize empty list for URLs
    queues = {}  # Initialize empty dictionary for queues
    threads = []  # Initialize a list to keep track of threads
    sender_task_ref = [None]  # Reference to the sender task
    
    try:
        # Start the receiver task
        receiver_task = asyncio.create_task(websocket_receiver(websocket, manager, queues, urls, sender_task_ref, threads))

        # Start the initial sender task if URLs are present
        if urls:
            sender_task_ref[0] = asyncio.create_task(websocket_sender(websocket, manager, queues, urls))

        # Wait for the receiver task to complete
        await receiver_task

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    finally:
        # Stop all threads
        stop_threads(threads)

        # Ensure sender task is also cancelled when the websocket is disconnected
        if sender_task_ref[0]:
            sender_task_ref[0].cancel()
            try:
                await sender_task_ref[0]  # Ensure the task is cancelled properly
            except asyncio.CancelledError:
                print("Sender task properly cancelled.")
        await manager.disconnect(websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)

if __name__ == "__main__": 
    origins = [
    "*"
    ]

    app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    import uvicorn
    # uvicorn.run(app, host="192.168.0.50", port=5555)
    uvicorn.run(app, host="0.0.0.0", port=5555)

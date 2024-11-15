from ultralytics import YOLO

def detect_bounding_boxes(yolo_model, image, target_classes=None, conf_threshold=0.2, iou_threshold=0.5):
    results = yolo_model(image, conf=conf_threshold, iou=iou_threshold, device=yolo_model.device, verbose=False)[0]
    bboxes = []
    for detection in results.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cls_idx = int(detection.cls)
        conf = float(detection.conf)
        label = results.names[cls_idx]
        if target_classes is None or cls_idx in target_classes:
            bboxes.append({
                'bbox': (x1, y1, x2, y2),
                'class': label,
                'confidence': conf
            })
    return bboxes

def get_bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y 
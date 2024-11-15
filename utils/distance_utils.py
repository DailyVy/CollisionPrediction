import math
import numpy as np

def calculate_distance(pos1, pos2):
    """두 3D 위치 간의 유클리드 거리 계산."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

def calculate_3d_distance(pos1, pos2, image_height=360):
    """
    CCTV 환경에 맞는 3D 거리 계산
    
    Args:
        pos1: (x, y, z) 첫 번째 위치
        pos2: (x, y, z) 두 번째 위치
        image_height: 이미지 세로 크기
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    
    # 두 점의 중간 y 위치를 기준으로 보정 계수 계산
    avg_y = (y1 + y2) / 2
    relative_position = avg_y / image_height
    position_correction = 1 + (1 - relative_position)
    
    # 테스트 결과 기준: 78.6% 지점에서 54.27 pixels/meter
    base_scale = 54.27  # 기준 픽셀 스케일
    base_position = 0.786  # 기준 위치 (78.6%)
    
    # 현재 위치에 따른 픽셀 스케일 조정
    position_diff = relative_position - base_position
    adjusted_scale = base_scale * (1 - position_diff)  # 위치 차이에 따른 스케일 조정
    
    # 2D 픽셀 거리 계산
    pixel_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # 픽셀 거리를 실제 거리(미터)로 변환
    real_distance = pixel_distance / adjusted_scale
    
    # depth 차이를 실제 거리로 변환 (0-255 범위를 0.1-10m 범위로 변환)
    depth_diff = abs(z2 - z1)
    depth_meters = depth_diff * (10.0 - 0.1) / 255.0
    
    # 3D 유클리드 거리 계산
    distance_3d = np.sqrt(real_distance**2 + depth_meters**2)
    
    return {
        'distance_3d': distance_3d,
        'pixel_distance': pixel_distance,
        'depth_difference': depth_meters,
        'weighted_score': distance_3d,
        'pixel_scale': adjusted_scale,
        'relative_position': relative_position,
        'position_correction': position_correction
    }

def create_distance_database(heavy_objects, persons, forklifts):
    """
    객체 감지 및 거리 정보를 구조화된 형태로 저장합니다.
    """
    db = {
        'frame_info': {
            'timestamp': None,
        },
        'objects': {
            'heavy_objects': [],
            'persons': [],
            'forklifts': []
        },
        'distances': []
    }
    
    # Heavy objects 정보 저장 (None이 아닐 경우에만)
    if heavy_objects:
        for idx, heavy_object in enumerate(heavy_objects):
            if isinstance(heavy_object, dict) and 'position_3d' in heavy_object:
                heavy_obj_data = {
                    'id': f'heavy_object_{idx}',
                    'position_2d': heavy_object['position'],
                    'position_3d': heavy_object['position_3d'],
                    'confidence': heavy_object['confidence'],
                    'bbox': heavy_object.get('bbox', None)
                }
                db['objects']['heavy_objects'].append(heavy_obj_data)
    
    # Person 정보 저장 및 거리 계산
    for idx, person in enumerate(persons):
        person_data = {
            'id': f'person_{idx}',
            'position_2d': person['position'],
            'position_3d': person['position_3d'],
            'confidence': person['confidence'],
            'bbox': person.get('bbox', None)
        }
        db['objects']['persons'].append(person_data)
        
        # Heavy objects가 있는 경우 각각에 대해 거리 계산
        for hidx, heavy_object in enumerate(heavy_objects):
            distance_info = calculate_3d_distance(
                heavy_object['position_3d'],
                person['position_3d']
            )
            db['distances'].append({
                'from_id': f'heavy_object_{hidx}',
                'to_id': f'person_{idx}',
                'distance_details': distance_info,
                'type': 'heavy_object_to_person'
            })
    
    # Forklift 정보 저장 및 거리 계산
    for idx, forklift in enumerate(forklifts):
        forklift_data = {
            'id': f'forklift_{idx}',
            'position_2d': forklift['position'],
            'position_3d': forklift['position_3d'],
            'confidence': forklift['confidence'],
            'bbox': forklift.get('bbox', None)
        }
        db['objects']['forklifts'].append(forklift_data)
        
        # Heavy objects가 있는 경우 각각에 대해 거리 계산
        for hidx, heavy_object in enumerate(heavy_objects):
            distance_info = calculate_3d_distance(
                heavy_object['position_3d'],
                forklift['position_3d']
            )
            db['distances'].append({
                'from_id': f'heavy_object_{hidx}',
                'to_id': f'forklift_{idx}',
                'distance_details': distance_info,
                'type': 'heavy_object_to_forklift'
            })
    
    return db

def print_distance_database(db):
    """데이터베이스 내용을 보기 좋게 출력합니다."""
    print("\n=== Object Detection and Distance Database ===")
    
    # Heavy Object 정보 출력
    if db['objects']['heavy_objects']:
        ho = db['objects']['heavy_objects']
        print("\nHeavy Objects:")
        for heavy_object in ho:
            print(f"\nHeavy Object: {heavy_object['id']}")
            print(f"Position 3D: (x: {heavy_object['position_3d'][0]:.2f}, y: {heavy_object['position_3d'][1]:.2f}, z: {heavy_object['position_3d'][2]:.2f}m)")
            print(f"Confidence: {heavy_object['confidence']:.3f}")
    
    # Person 정보 출력
    if db['objects']['persons']:
        print("\nPersons:")
        for person in db['objects']['persons']:
            print(f"\n{person['id'].upper()}:")
            print(f"Position 3D: (x: {person['position_3d'][0]:.2f}, y: {person['position_3d'][1]:.2f}, z: {person['position_3d'][2]:.2f}m)")
            print(f"Confidence: {person['confidence']:.3f}")
    
    # Forklift 정보 출력
    if db['objects']['forklifts']:
        print("\nForklifts:")
        for forklift in db['objects']['forklifts']:
            print(f"\n{forklift['id'].upper()}:")
            print(f"Position 3D: (x: {forklift['position_3d'][0]:.2f}, y: {forklift['position_3d'][1]:.2f}, z: {forklift['position_3d'][2]:.2f}m)")
            print(f"Confidence: {forklift['confidence']:.3f}")
    
    # 거리 정보 출력
    if db['distances']:
        print("\nDistances:")
        for dist in db['distances']:
            print(f"{dist['from_id']} to {dist['to_id']}: {dist['distance']:.2f}m")

def calculate_pixel_scale_from_person(person_bbox, image_height=360, real_height_cm=170):
    """
    사람의 bbox와 실제 키를 기준으로 픽셀 스케일 계산 (위치 기반 보정 포함)
    
    Args:
        person_bbox: (x_min, y_min, x_max, y_max) 형태의 bbox
        image_height: 이미지 세로 크기
        real_height_cm: 실제 키 (cm)
    """
    x_min, y_min, x_max, y_max = person_bbox
    bbox_height_pixels = y_max - y_min
    
    # 화면에서의 상대적 위치 계산 (0: 맨 위, 1: 맨 아래)
    relative_position = y_max / image_height
    
    # 위치 기반 보정 계수 계산
    # 예: 화면 아래쪽(1.0)은 보정 없음, 화면 위쪽(0.0)은 최대 2배까지 보정
    position_correction = 1 + (1 - relative_position)
    
    # cm를 m로 변환
    real_height_m = real_height_cm / 100
    
    # 보정된 픽셀 스케일 계산
    pixels_per_meter = (bbox_height_pixels * position_correction) / real_height_m
    
    return {
        'pixels_per_meter': pixels_per_meter,
        'relative_position': relative_position,
        'correction_factor': position_correction,
        'original_pixels_per_meter': bbox_height_pixels / real_height_m
    }


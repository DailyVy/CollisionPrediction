import math
import numpy as np

def calculate_distance(pos1, pos2):
    """두 3D 위치 간의 유클리드 거리 계산."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

def calculate_3d_distance(pos1, pos2, w_pixel=1.0, w_depth=2.0):
    """
    CCTV 환경에 적합한 3D 거리를 계산합니다.
    2D 픽셀 거리와 깊이 차이를 별도로 계산하고 가중치를 적용합니다.
    
    Args:
        pos1: (x, y, z) 첫 번째 위치 (x,y는 픽셀, z는 미터)
        pos2: (x, y, z) 두 번째 위치 (x,y는 픽셀, z는 미터)
        w_pixel: 픽셀 거리에 대한 가중치 (기본값: 1.0)
        w_depth: 깊이 차이에 대한 가중치 (기본값: 2.0)
    
    Returns:
        dict: {
            'total_distance': 종합적인 거리 점수,
            'pixel_distance': 2D 픽셀 거리,
            'depth_difference': 깊이 차이(미터),
            'weighted_score': 가중치가 적용된 최종 점수
        }
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    
    # 2D 픽셀 거리 계산
    pixel_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # 깊이 차이 계산 (미터)
    depth_difference = abs(z2-z1)
    
    # 픽셀 거리를 대략적인 미터 단위로 정규화 (예시 값, 실제 환경에 맞게 조정 필요)
    # 예: 1920x1080 해상도에서 1미터가 약 100픽셀이라고 가정
    normalized_pixel_distance = pixel_distance / 100.0
    
    # 가중치가 적용된 최종 거리 점수 계산
    weighted_score = (w_pixel * normalized_pixel_distance + w_depth * depth_difference) / (w_pixel + w_depth)
    
    return {
        'total_distance': weighted_score,
        'pixel_distance': pixel_distance,
        'depth_difference': depth_difference,
        'weighted_score': weighted_score
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
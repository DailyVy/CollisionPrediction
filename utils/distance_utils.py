import math
import numpy as np

def calculate_distance(pos1, pos2):
    """두 3D 위치 간의 유클리드 거리 계산."""
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) 

def calculate_3d_distance(point1, point2):
    """
    CCTV에서 감지된 두 객체 간의 3D 거리를 계산합니다.
    
    Parameters:
    point1: dict containing {'x': pixel_x, 'y': pixel_y, 'depth': relative_depth}
    point2: dict containing {'x': pixel_x, 'y': pixel_y, 'depth': relative_depth}
    
    Returns:
    float: 두 점 사이의 3D 거리
    """
    # 2D 픽셀 거리 계산
    pixel_distance = np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
    
    # 깊이 차이 계산
    depth_difference = abs(point1['depth'] - point2['depth'])
    
    # 3D 거리 계산 (피타고라스 정리 활용)
    distance_3d = np.sqrt(pixel_distance**2 + depth_difference**2)
    
    return distance_3d
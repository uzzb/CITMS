import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple
import asyncio

from traffic_management.models import AnalysisRecord, TrafficAnalysisRecord
from traffic_management.services.frame_processor import FrameProcessor
from traffic_management.base_types import AnalysisData


class EnhancedFrameProcessor(FrameProcessor):
    # region 初始化
    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                 location_id: int = None, location_name: str = None,
                 density_grid_size: int = 25):
        super().__init__(conf_threshold, iou_threshold)
        self.location_id = location_id
        self.location_name = location_name
        self.save_interval = 5.0
        self.last_save_time = 0

        # 基础参数设置
        self.density_grid_size = density_grid_size
        self.tracking_distance_threshold = 30
        self.tracking_history_length = 10

        # 阈值设置
        self.crowd_threshold = 0.3
        self.velocity_threshold = 1.5
        self.density_threshold = 0.3

        # 交通分析参数
        self.flow_threshold = 0.3  # 判定为主要流向的阈值
        self.bottleneck_threshold = 0.7  # 判定为瓶颈点的阈值
        self.congestion_threshold = 0.8  # 判定为拥堵的阈值

        # 定义车道区域和容量
        self.lane_areas = {
            'lane1': np.array([  # 南三国旗杆台阶区域
                [150, 250], [450, 250],  # 上边
                [500, 400], [100, 400]  # 下边
            ]),
            'lane2': np.array([  # 人行道区域
                [100, 100], [500, 100],  # 上边
                [500, 250], [100, 250]  # 下边
            ]),
            'lane3': np.array([  # 绿化带区域
                [500, 100], [600, 100],  # 上边
                [600, 400], [500, 400]  # 下边
            ])
        }

        self.lane_capacities = {
            'lane1': 100,  # 台阶区域容量
            'lane2': 150,  # 人行道容量
            'lane3': 50  # 绿化带容量
        }

        # 初始化追踪器和计数器
        self.tracks = {}
        self.next_track_id = 0
        self.zone_counts = defaultdict(int)
        self.zone_densities = {}

        # 交通分析数据存储
        self.flow_history = defaultdict(list)  # 存储历史流向
        self.speed_history = defaultdict(list)  # 存储历史速度
        self.bottleneck_history = []  # 存储瓶颈点历史
    # endregion

    # region 车

    # region 基础工具方法
    def _point_in_area(self, point: tuple, area: np.ndarray) -> bool:
        """
        判断点是否在多边形区域内

        Args:
            point: 点坐标 (x, y)
            area: 多边形区域顶点坐标数组 [[x1,y1], [x2,y2], ...]

        Returns:
            bool: 点是否在区域内
        """
        x, y = point
        n = len(area)
        inside = False

        j = n - 1
        for i in range(n):
            if ((area[i][1] > y) != (area[j][1] > y) and
                    (x < (area[j][0] - area[i][0]) * (y - area[i][1]) /
                     (area[j][1] - area[i][1]) + area[i][0])):
                inside = not inside
            j = i

        return inside

    def _angle_to_direction(self, angle: float) -> str:
        """
        将角度转换为方向描述

        Args:
            angle: 弧度角度

        Returns:
            str: 方向描述
        """
        # 转换为度数并确保在0-360范围内
        degrees = (np.degrees(angle) + 360) % 360

        # 定义方向区间
        directions = {
            (337.5, 22.5): "北",
            (22.5, 67.5): "东北",
            (67.5, 112.5): "东",
            (112.5, 157.5): "东南",
            (157.5, 202.5): "南",
            (202.5, 247.5): "西南",
            (247.5, 292.5): "西",
            (292.5, 337.5): "西北"
        }

        # 确定方向
        for (start, end), direction in directions.items():
            if start > end:  # 处理跨越0度的情况
                if degrees >= start or degrees < end:
                    return direction
            else:
                if start <= degrees < end:
                    return direction

        return "北"  # 默认返回北

    def _calculate_velocity(self, track: dict) -> float:
        """
        计算轨迹的速度

        Args:
            track: 轨迹数据字典

        Returns:
            float: 速度值
        """
        if len(track['history']) < 2:
            return 0.0

        # 取最近两个点计算速度
        p1 = np.array(track['history'][-2])
        p2 = np.array(track['history'][-1])
        return float(np.linalg.norm(p2 - p1))

    def _calculate_saturation(self, count: int, lane_id: str) -> float:
        """
        计算车道饱和度

        Args:
            count: 当前车道内的对象数量
            lane_id: 车道ID

        Returns:
            float: 饱和度 [0,1]
        """
        capacity = self.lane_capacities.get(lane_id, 100)
        return min(1.0, count / capacity)

    # endregion

    # region 速度分析相关
    def _analyze_speed_distribution(self) -> dict:
        """分析速度分布情况"""
        if not self.tracks or not any(len(track['history']) >= 2 for track in self.tracks.values()):
            return {
                'histogram': {},
                'mean': 0.0,
                'std': 0.0,
                'percentiles': {'25': 0.0, '50': 0.0, '75': 0.0},
                'speed_classes': {'slow': 0, 'medium': 0, 'fast': 0}
            }

        # 计算所有轨迹的速度
        velocities = []
        for track in self.tracks.values():
            if len(track['history']) >= 2:
                velocity = self._calculate_velocity(track)
                if velocity > 0:  # 排除静止对象
                    velocities.append(velocity)

        if not velocities:
            return {
                'histogram': {},
                'mean': 0.0,
                'std': 0.0,
                'percentiles': {'25': 0.0, '50': 0.0, '75': 0.0},
                'speed_classes': {'slow': 0, 'medium': 0, 'fast': 0}
            }

        velocities = np.array(velocities)

        # 计算直方图
        hist, bin_edges = np.histogram(velocities, bins=10)
        histogram = {
            f"{float(bin_edges[i]):.1f}-{float(bin_edges[i + 1]):.1f}": float(hist[i])
            for i in range(len(hist))
        }

        return {
            'histogram': histogram,
            'mean': float(np.mean(velocities)),
            'std': float(np.std(velocities)),
            'percentiles': {
                '25': float(np.percentile(velocities, 25)),
                '50': float(np.percentile(velocities, 50)),
                '75': float(np.percentile(velocities, 75))
            },
            'speed_classes': self._classify_speeds(velocities)
        }

    def _classify_speeds(self, velocities: np.ndarray) -> dict:
        """
        将速度分类为慢、中、快三个等级

        Args:
            velocities: 速度数组

        Returns:
            dict: 各等级的数量统计
        """
        # 使用四分位数来定义速度等级
        q1 = np.percentile(velocities, 25)
        q3 = np.percentile(velocities, 75)

        # 统计各等级数量
        slow_count = int(np.sum(velocities <= q1))
        fast_count = int(np.sum(velocities >= q3))
        medium_count = int(len(velocities) - slow_count - fast_count)

        return {
            'slow': slow_count,
            'medium': medium_count,
            'fast': fast_count
        }

    def _get_speed_thresholds(self) -> tuple:
        """
        获取速度阈值设置

        Returns:
            tuple: (慢速阈值, 快速阈值)
        """
        # 这里的阈值可以根据实际场景调整
        slow_threshold = 0.5  # 米/秒
        fast_threshold = 2.0  # 米/秒
        return slow_threshold, fast_threshold

    def _calculate_speed_metrics(self) -> dict:
        """
        计算速度相关的指标

        Returns:
            dict: 速度指标统计
        """
        if not self.tracks:
            return {
                'average_speed': 0.0,
                'max_speed': 0.0,
                'speed_variation': 0.0
            }

        speeds = []
        for track in self.tracks.values():
            if len(track['history']) >= 2:
                speed = self._calculate_velocity(track)
                speeds.append(speed)

        if not speeds:
            return {
                'average_speed': 0.0,
                'max_speed': 0.0,
                'speed_variation': 0.0
            }

        speeds = np.array(speeds)

        return {
            'average_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'speed_variation': float(np.std(speeds))
        }
    # endregion

    # region 交通流分析
    def _analyze_traffic_patterns(self) -> dict:
        """分析交通模式"""
        if not self.tracks:
            return {}

        # 1. 分析流向
        flow_vectors = self._calculate_flow_vectors()
        main_directions = self._identify_main_directions(flow_vectors)

        # 2. 分析车道占用
        lane_occupancy = self._calculate_lane_occupancy()

        # 3. 分析速度分布
        speed_stats = self._analyze_speed_distribution()

        # 4. 检测瓶颈和拥堵
        bottlenecks = self._detect_bottlenecks()

        # 确保所有数据都是JSON可序列化的
        return {
            'flow_vectors': flow_vectors.tolist() if isinstance(flow_vectors, np.ndarray) else [],
            'main_directions': main_directions,
            'lane_occupancy': lane_occupancy,
            'speed_stats': speed_stats,
            'bottlenecks': bottlenecks
        }

    def _calculate_flow_vectors(self) -> np.ndarray:
        """计算流向向量"""
        flow_vectors = []
        for track in self.tracks.values():
            if len(track['history']) >= 5:  # 至少需要5个历史点
                start = np.array(track['history'][0])
                end = np.array(track['history'][-1])
                vector = end - start
                if np.linalg.norm(vector) > 20:  # 最小移动距离阈值
                    flow_vectors.append(vector.tolist())  # 转换为列表
        return np.array(flow_vectors)

    def _identify_main_directions(self, flow_vectors: np.ndarray) -> List[dict]:
        """识别主要流向"""
        if len(flow_vectors) < 2:
            return []

        # 计算角度
        angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])

        # 使用直方图统计主要方向
        hist, bins = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        main_dirs = []

        # 找出主要方向
        threshold = float(np.mean(hist) + np.std(hist))  # 转换为Python float
        for i, count in enumerate(hist):
            if count > threshold:
                angle = float((bins[i] + bins[i + 1]) / 2)  # 转换为Python float
                main_dirs.append({
                    'angle': angle,
                    'count': int(count),  # 转换为Python int
                    'direction': self._angle_to_direction(angle)
                })

        return main_dirs

    def _calculate_lane_occupancy(self) -> dict:
        """计算车道占用率"""
        occupancy_data = {}
        for lane_id, lane_area in self.lane_areas.items():
            objects_in_lane = sum(1 for track in self.tracks.values()
                                  if self._point_in_area(track['position'], lane_area))
            occupancy_data[lane_id] = {
                'count': int(objects_in_lane),  # 确保是Python int
                'occupancy_rate': float(objects_in_lane / self.lane_capacities.get(lane_id, 100)),  # 确保是Python float
                'saturation': float(self._calculate_saturation(objects_in_lane, lane_id))  # 确保是Python float
            }
        return occupancy_data

    def _detect_bottlenecks(self) -> List[dict]:
        """检测交通瓶颈"""
        if len(self.tracks) < 5:
            return []

        try:
            positions = np.array([track['position'] for track in self.tracks.values()])
            velocities = np.array([self._calculate_velocity(track) for track in self.tracks.values()])

            # 使用KDE检测密度热点
            from scipy.stats import gaussian_kde
            kernel = gaussian_kde(positions.T)
            density = kernel(positions.T)

            # 找出高密度低速度的区域
            bottleneck_indicators = density * (1 / (velocities + 1e-6))
            threshold = float(np.percentile(bottleneck_indicators, 90))  # 转换为Python float

            bottlenecks = []
            for i, (pos, indicator) in enumerate(zip(positions, bottleneck_indicators)):
                if indicator > threshold:
                    bottlenecks.append({
                        'position': [float(x) for x in pos],  # 转换为Python float列表
                        'severity': float(indicator),
                        'velocity': float(velocities[i])
                    })

            return bottlenecks
        except Exception as e:
            print(f"Error in bottleneck detection: {e}")
            return []

    #     endregion
    # endregion

    #region 人
    def _create_gaussian_kernel(self, size=3, sigma=1.0):
        """创建高斯核"""
        x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
        d = np.sqrt(x * x + y * y)
        gaussian_kernel = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        return gaussian_kernel / gaussian_kernel.sum()

    def _calculate_density(self, positions: List[Tuple[int, int]], frame_shape: Tuple[int, ...]) -> float:
        """
        计算区域密度

        Args:
            positions: 位置坐标列表 [(x1,y1), (x2,y2),...]
            frame_shape: 帧的形状 (height, width)

        Returns:
            float: 归一化的密度值 [0,1]
        """
        if not positions:
            return 0.0

        height, width = frame_shape[:2]
        grid_h = height // self.density_grid_size + 1
        grid_w = width // self.density_grid_size + 1

        # 创建密度图，确保大小足够容纳kernel
        density_map = np.zeros((grid_h + 2, grid_w + 2))

        # 创建高斯核
        kernel_size = 3
        gaussian_kernel = self._create_gaussian_kernel(kernel_size)

        # 计算每个网格的最大容纳人数（用于归一化）
        max_people_per_grid = (self.density_grid_size ** 2) / (1.0 ** 2)  # 假设每人占用1平方米

        # 累积位置
        for x, y in positions:
            grid_x = int(x // self.density_grid_size)
            grid_y = int(y // self.density_grid_size)

            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                # 计算有效的网格范围
                grid_y_start = max(0, grid_y)
                grid_y_end = min(grid_h + 2, grid_y + kernel_size)
                grid_x_start = max(0, grid_x)
                grid_x_end = min(grid_w + 2, grid_x + kernel_size)

                # 计算对应的核范围
                kernel_y_start = max(0, -grid_y)
                kernel_y_end = min(kernel_size, grid_h + 2 - grid_y)
                kernel_x_start = max(0, -grid_x)
                kernel_x_end = min(kernel_size, grid_w + 2 - grid_x)

                # 确保kernel片段和目标区域大小匹配
                kernel_slice = gaussian_kernel[kernel_y_start:kernel_y_end,
                               kernel_x_start:kernel_x_end]
                target_slice = density_map[grid_y_start:grid_y_end,
                               grid_x_start:grid_x_end]

                if kernel_slice.shape == target_slice.shape:
                    density_map[grid_y_start:grid_y_end,
                    grid_x_start:grid_x_end] += kernel_slice

        # 计算实际密度
        if len(positions) == 0:
            return 0.0

        # 计算有效区域
        valid_area = grid_h * grid_w
        people_density = len(positions) / valid_area

        # 归一化密度值
        normalized_density = min(people_density / (max_people_per_grid * 0.3), 1.0)

        # 确保密度值在[0,1]范围内
        normalized_density = max(0.0, min(1.0, normalized_density))

        return float(normalized_density)

    def _detect_abnormal_events(self) -> List[str]:
        """检测异常事件"""
        events = []
        if not self.tracks:
            return events

        # 计算速度和加速度
        velocities = []
        accelerations = []
        positions = []

        for track in self.tracks.values():
            if len(track['history']) >= 3:
                # 计算速度
                p1 = np.array(track['history'][-2])
                p2 = np.array(track['history'][-1])
                velocity = np.linalg.norm(p2 - p1)
                velocities.append(velocity)

                # 计算加速度
                p0 = np.array(track['history'][-3])
                v1 = np.linalg.norm(p1 - p0)
                v2 = np.linalg.norm(p2 - p1)
                acceleration = abs(v2 - v1)
                accelerations.append(acceleration)

                positions.append(p2)

        if velocities:
            # 检测快速移动
            mean_velocity = np.mean(velocities)
            if mean_velocity > self.velocity_threshold:
                events.append('快速移动')

            # 检测突然加速/减速
            if accelerations:
                mean_acceleration = np.mean(accelerations)
                if mean_acceleration > self.velocity_threshold * 0.5:
                    events.append('突然加速')

        # 检测拥挤情况
        if positions:
            positions = np.array(positions)
            if len(positions) >= 3:
                # 计算局部密度
                from scipy.spatial import cKDTree
                tree = cKDTree(positions)
                distances, _ = tree.query(positions, k=3)
                mean_distances = np.mean(distances, axis=1)

                # 如果平均距离小于阈值，认为是拥挤
                if np.mean(mean_distances) < self.density_threshold * 50:
                    events.append('拥挤')

                # 检测异常聚集
                std_distances = np.std(mean_distances)
                if std_distances < self.density_threshold * 10:
                    events.append('异常聚集')

        return events

    def _update_tracks(self, detections: torch.Tensor, frame_id: int):
        """更新目标追踪"""
        current_detections = []

        # 过滤和处理检测结果
        for det in detections:
            if det[5] != 0:  # 只跟踪人员(类别0)
                continue

            bbox = det[:4].cpu().numpy()
            confidence = det[4].item()

            if confidence < self.conf_threshold:
                continue

            # 计算边界框中心点
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            current_detections.append((center, confidence))

        # 使用匈牙利算法进行追踪匹配
        if self.tracks and current_detections:
            from scipy.optimize import linear_sum_assignment

            # 构建成本矩阵
            cost_matrix = np.zeros((len(self.tracks), len(current_detections)))
            track_ids = list(self.tracks.keys())

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for j, (center, _) in enumerate(current_detections):
                    distance = np.linalg.norm(np.array(center) - np.array(track['position']))
                    cost_matrix[i, j] = distance

            # 使用匈牙利算法进行匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # 更新匹配的轨迹
            for row_idx, col_idx in zip(row_indices, col_indices):
                track_id = track_ids[row_idx]
                center, conf = current_detections[col_idx]

                # 只更新距离在阈值内的轨迹
                if cost_matrix[row_idx, col_idx] < self.tracking_distance_threshold:
                    self.tracks[track_id]['position'] = center
                    self.tracks[track_id]['last_frame'] = frame_id

                    # 更新轨迹历史，保持固定长度
                    if len(self.tracks[track_id]['history']) > self.tracking_history_length:
                        self.tracks[track_id]['history'].pop(0)
                    self.tracks[track_id]['history'].append(center)
                else:
                    # 距离过大，删除轨迹
                    del self.tracks[track_id]

            # 删除未匹配的轨迹
            matched_tracks = set(track_ids[i] for i in row_indices)
            for track_id in list(self.tracks.keys()):
                if track_id not in matched_tracks:
                    del self.tracks[track_id]

            # 添加未匹配的检测为新轨迹
            matched_detections = set(col_indices)
            for i, (center, conf) in enumerate(current_detections):
                if i not in matched_detections:
                    self.tracks[self.next_track_id] = {
                        'position': center,
                        'history': [center],
                        'last_frame': frame_id,
                        'id': self.next_track_id
                    }
                    self.next_track_id += 1

        else:
            # 如果没有现有轨迹，将所有检测创建为新轨迹
            for center, conf in current_detections:
                self.tracks[self.next_track_id] = {
                    'position': center,
                    'history': [center],
                    'last_frame': frame_id,
                    'id': self.next_track_id
                }
                self.next_track_id += 1

    # endregion

    # region 共同的处理
    async def process_frame(self, frame: np.ndarray, model: torch.nn.Module) -> Tuple[np.ndarray, AnalysisData]:
        try:
            with torch.no_grad():
                results = model(frame)

            processed_frame = await super().process_frame(frame, model)
            frame_id = int(cv2.getTickCount())
            self._update_tracks(results.xyxy[0], frame_id)

            positions = [track['position'] for track in self.tracks.values()]
            crowd_density = self._calculate_density(positions, frame.shape)
            abnormal_events = self._detect_abnormal_events()

            # 计算平均速度和加速度
            mean_velocity = 0.0
            mean_acceleration = 0.0
            if self.tracks:
                velocities = []
                accelerations = []
                for track in self.tracks.values():
                    if len(track['history']) >= 3:
                        p1 = np.array(track['history'][-2])
                        p2 = np.array(track['history'][-1])
                        velocity = np.linalg.norm(p2 - p1)
                        velocities.append(velocity)

                        p0 = np.array(track['history'][-3])
                        v1 = np.linalg.norm(p1 - p0)
                        acceleration = abs(velocity - v1)
                        accelerations.append(acceleration)

                if velocities:
                    mean_velocity = float(np.mean(velocities))
                if accelerations:
                    mean_acceleration = float(np.mean(accelerations))

            traffic_analysis = self._analyze_traffic_patterns()

            # 创建分析数据
            analysis_data = AnalysisData(
                location_id=self.location_id,
                zone_counts={},  # 暂时使用空字典
                warnings=set(abnormal_events),
                crowd_density=float(crowd_density),
                total_count=len(self.tracks),
                velocity=mean_velocity,
                abnormal_events=abnormal_events,
                timestamp=datetime.now(),
                acceleration=mean_acceleration,
                zone_densities={}, # 暂时使用空字典
                traffic_analysis=traffic_analysis,
            )

            # 保存分析数据
            try:
                current_time = asyncio.get_event_loop().time()
                if (current_time - self.last_save_time) >= self.save_interval:
                    await asyncio.to_thread(
                        AnalysisRecord.save_analysis,
                        self.location_id,
                        analysis_data
                    )
                    await asyncio.to_thread(
                        TrafficAnalysisRecord.save_analysis,
                        self.location_id,
                        traffic_analysis
                    )
                    self.last_save_time = current_time
            except Exception as e:
                traceback.print_exc()



            # 可视化结果
            processed_frame = self._visualize_results(frame, analysis_data)

            return processed_frame, analysis_data

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame, None

    def _visualize_results(self, frame: np.ndarray, analysis_data: AnalysisData) -> np.ndarray:
        """可视化结果，添加位置信息"""
        # 添加位置信息
        if self.location_name:
            cv2.putText(frame, f"Location: {self.location_name}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 1. 绘制轨迹
        for track in self.tracks.values():
            if len(track['history']) > 1:
                points = np.array(track['history'], dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)

                # 预测路径
                if len(points) >= 3:
                    try:
                        x = np.arange(len(points))
                        px = np.polyfit(x, points[:, 0], 1)
                        py = np.polyfit(x, points[:, 1], 1)
                        next_x = int(px[0] * (len(points) + 1) + px[1])
                        next_y = int(py[0] * (len(points) + 1) + py[1])
                        cv2.line(frame, tuple(points[-1]), (next_x, next_y),
                                 (255, 0, 255), 1, cv2.LINE_AA)
                    except Exception:
                        pass

        # 2. 显示基本统计信息
        y_offset = 50
        info_color = (255, 255, 255)

        # 显示基本信息
        basic_info = [
            f"Total Count: {analysis_data.total_count}",
            f"Density: {analysis_data.crowd_density:.2f}",
        ]
        if analysis_data.velocity > 0:
            basic_info.append(f"Avg Speed: {analysis_data.velocity:.2f}")

        for info in basic_info:
            cv2.putText(frame, info, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
            y_offset += 20

        # 3. 显示警告
        if analysis_data.warnings:
            cv2.putText(frame, "Warnings:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20

            for warning in analysis_data.warnings:
                cv2.putText(frame, f"- {warning}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 20

        # 4. 显示交通分析信息
        if hasattr(analysis_data, 'traffic_analysis'):
            traffic_data = analysis_data.traffic_analysis

            # 绘制主要流向
            if 'main_directions' in traffic_data:
                for direction in traffic_data['main_directions']:
                    angle = direction['angle']
                    strength = direction['count']
                    center = np.array([frame.shape[1] // 2, frame.shape[0] // 2])
                    end = center + np.array([
                        np.cos(angle) * strength,
                        np.sin(angle) * strength
                    ]) * 50
                    cv2.arrowedLine(frame,
                                    tuple(center.astype(int)),
                                    tuple(end.astype(int)),
                                    (0, 255, 0), 2)

            # 标记瓶颈点
            if 'bottlenecks' in traffic_data:
                for bottleneck in traffic_data['bottlenecks']:
                    pos = tuple(map(int, bottleneck['position']))
                    severity = bottleneck['severity']
                    color = (0, 0, 255) if severity > 0.7 else (0, 255, 255)
                    cv2.circle(frame, pos, 10, color, -1)

            # 显示车道占用率
            base_y_offset = y_offset + 20
            if 'lane_occupancy' in traffic_data:
                cv2.putText(frame, "Lane Occupancy:", (10, base_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
                base_y_offset += 20

                for lane_id, data in traffic_data['lane_occupancy'].items():
                    occupancy_text = f"{lane_id}: {data['occupancy_rate']:.1%} occupied"
                    cv2.putText(frame, occupancy_text,
                                (20, base_y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                info_color,
                                1)
                    base_y_offset += 20

        return frame

    # endregion
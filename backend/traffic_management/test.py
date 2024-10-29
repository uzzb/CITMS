import json
import os

import cv2
import torch
import asyncio
import numpy as np
from collections import deque
from channels.generic.websocket import AsyncWebsocketConsumer
import torchvision
from datetime import datetime
from natsort import natsorted

from traffic_management.services.db_services import insert_traffic_data

'''
    当WebSocket连接建立时，调用connect()函数。可以在其中进行身份认证或者初始化
    之后，如果由消息接收则会调用receive函数。可以在这里处理发来的数据。
    最后关闭WebSocket时调用disconnect
'''


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    '''
        即使发送空消息，video_url就会有下述信息
    '''
    async def receive(self, text_data):
        data = json.loads(text_data)
        directory = r'E:\道路视频\南三国旗杆台阶_20241015174856'

        video_files = []
        for root, dirs, files in os.walk(directory):
            for file in natsorted(files):
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

        # 每次只读取一个文件进行处理
        for idx, video in enumerate(video_files):
            await self.process_stream(video,stream_id = idx+1)

    async def process_stream(self, video_url, stream_id):
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                await self.send(text_data=json.dumps({'error': 'Unable to open video URL'}))
                return

            model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            model = model.to('cuda')
            model.eval()
            fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

            detection_interval = 5
            frame_rate = 30
            frame_interval = 1.0 / frame_rate
            trajectories = {}
            frame_count = 0
            lane_boundaries = [480, 960, 1440, 1920]
            rush_hour_ranges = [('07:00:00', '09:00:00')], ('17:00:00', '19:00:00')

            traffic_data = {
                'vehicle_count': 0,
                'people_count': 0,
                'vehicle_speeds': [],
                'vehicle_directions': [],
                'people_directions': [],
                'incident_count': 0,
                'people_density':  0,
                'vehicle_density': 0,
                'flow_matrix': np.zeros((4,4)), # 流量矩阵的 4x4 网格示例
                'lane_occupancy': {}
            }

            crowd_paths = {}
            activity_data = {}
            historical_density_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                fgmask = fgbg.apply(frame)

                # 每5帧进行一次检测
                if frame_count % detection_interval == 0:
                    '''
                        将frame转换为frame_tensor实际上是将numpy数组转换为PyTorch张量
                        将numpy数组直接输入到模型中会导致错误。
                        frame_tensor可以利用GPU加速进行推理，从而提高处理速度。
                        而使用frame会导致性能下降，因为CPU处理图像数据的速度相对较慢。
                    '''
                    frame_tensor = torch.from_numpy(frame).float().cuda()
                    if len(frame_tensor.shape) == 3:
                        frame_tensor = frame_tensor.permute(2,0,1)
                    frame_tensor = frame_tensor.unsqueeze(0)

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            results = model(frame_tensor)

                    # conf_threshold = 0.5
                    iou_threshold = 0.4
                    detections = results[0]
                    # 过滤多余的边框，确保检测到的对象都由一个准确的边框表示
                    # 使用PyTorch生态系统中的一个软件包torchvision库。
                    # torchvision.ops.nms 非最大抑制的缩写。减少物体检测中重叠边界框数量。
                    # detections[:,:4] 检测数组中提取包围盒的坐标
                    # detections[:,4] 提取检测结果的置信度分数
                    # iou_threshold 交集大于联合的阈值，决定了两个边界框可接受的重叠程度，若大于它则其中一个将被抑制。
                    keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_threshold)
                    # 用处理过后的边界框覆盖原有的。
                    detections = detections[keep]

                    trajectories.clear()

                    for detection in detections:
                        x1, y1, x2, y2, conf, cls = detection[:6].tolist()
                        # if conf < conf_threshold:
                        #     continue
                        label = model.names[int(cls)]
                        if label in ['person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus']:
                            center_y, center_x = int((y1 + y2) / 2), int((x1 + x2) / 2)
                            # 检查边界框中心是否位于前景掩码范围内。
                            if fgmask[center_y, center_x] > 0:
                                width, height = x2 - x1, y2 - y1
                                if width > 0 and height > 0:
                                    bbox = (float(x1), float(y1), width, height)
                                    # 创建KCF(核化相关滤波器)跟踪器，并使用当前帧和边界框进行初始化。用于在之后帧中跟踪物体。
                                    if frame is not None:
                                        tracker = cv2.legacy.TrackerKCF_create()
                                        tracker.init(frame, bbox)
                                        # 初始化一个字典，用于存储每个对象的跟踪信息。
                                        trajectories[len(trajectories)] = {
                                            'tracker': tracker,
                                            'label': label,
                                            'points': deque(maxlen=30),
                                            'bbox': bbox,
                                            'speed': 0,
                                            'direction': None,
                                            'speeds': [],
                                            'directions': []
                                        }
                                    else:
                                        print("Warning: Frame is None, skipping tracker initialization.")
                                else:
                                    print("Warning: Invalid bbox dimensions, skipping tracker initialization.")

                people_count = 0
                vehicle_count = 0
                frame_area = frame.shape[0] * frame.shape[1]

                # 轨迹字典中的键和键对应的值
                for track_id, traj in trajectories.items():
                    success, bbox = traj['tracker'].update(frame)
                    if success:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        center_y, center_x = int((p1[1] + p2[1]) / 2), int((p1[0] + p2[0]) / 2)

                        if fgmask[center_y, center_x] > 0:
                            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                            cv2.putText(frame, traj['label'], (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 2)

                            if len(traj['points']) >= 4:

                                # 根据像素点的变化计算速度，并测得方向。
                                dx = p1[0] - traj['points'][-4][0]
                                dy = p1[1] - traj['points'][-4][1]
                                speed = np.sqrt(dx ** 2 + dy ** 2)
                                direction = np.arctan2(dy,dx)

                                traj['speed'] = speed
                                traj['direction'] = direction

                                cv2.putText(frame, f'Speed: {speed:.2f}', (p1[0], p1[1] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.9, (255, 255, 0), 2)

                                if traj['label'] == 'person':
                                    traffic_data['people_directions'].append(direction)
                                    crowd_paths[track_id] = traj['points']  # For behavior analysis
                                else:
                                    traffic_data['vehicle_speeds'].append(speed)
                                    traffic_data['vehicle_directions'].append(direction)
                                    update_flow_matrix(traffic_data,traj)

                            traj['points'].append(p1)

                            if traj['label'] == 'person':
                                people_count += 1
                            elif traj['label'] in ['car', 'bus', 'truck']:
                                vehicle_count += 1
                                # 车道占用率
                                lane_id = determine_lane(traj['bbox'])
                                traffic_data['lane_occupancy'].setdefault(lane_id, 0)
                                traffic_data['lane_occupancy'][lane_id] += 1

                people_density = people_count / frame_area
                vehicle_density = vehicle_count / frame_area

                # traffic_data['people_count'] += people_count
                # traffic_data['vehicle_count'] += vehicle_count
                traffic_data['people_density'] = people_density
                traffic_data['vehicle_density'] = vehicle_density
                historical_density_data.append(people_density + vehicle_density)

                # 高密度拥堵检测
                if people_density > 0.05: # 需根据具体情况调整阈值
                    cv2.putText(frame, 'Crowded Area Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if vehicle_density > 0.05:  # 同上
                    cv2.putText(frame, 'Traffic Congestion Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                detect_abnormal_behavior(traffic_data, crowd_paths)

                current_time = datetime.now().strftime("%H:%M:%S")
                analyze_activity_by_time(current_time, traffic_data, activity_data)

                handle_incidents(traffic_data, traj, frame)
                handle_abnormal_behaviors(traffic_data, traj, frame)

                # cv2.putText(frame, f'PDensity: {people_density:.6f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (255, 255, 255), 2)
                # cv2.putText(frame, f'VDensity: {vehicle_density:.6f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (255, 255, 255), 2)
                #
                # parking_zones = [(x1, y1, x2, y2), ...]
                #
                # for traj in trajectories.values():
                #     if traj['label'] in ['car', 'bus', 'truck']:
                #         if traj['speed'] < 0.5:
                #             if is_in_non_parking_zone(traj['bbox'], parking_zones):
                #                 traffic_data['incident_count'] += 1
                #                 cv2.putText(frame, 'Parking Violation', (p1[0], p1[1] - 30),
                #                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                #
                # speed_limit = 50
                # for traj in trajectories.values():
                #     if traj['speed'] > speed_limit:
                #         traffic_data['incident_count'] += 1
                #         cv2.putText(frame, 'Speed Violation', (p1[0], p1[1] - 50,
                #                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2))
                #
                # for direction in traffic_data['vehicle_directions']:
                #     if abs(direction) > np.pi / 2:
                #         traffic_data['incident_count'] += 1
                #         cv2.putText(frame, 'Wrong way Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1 ,(0,0,255),2)

                if frame_count % (detection_interval * 30) == 0:
                    insert_traffic_data(stream_id, traffic_data)

                frame = visualize_flow_matrix(frame, traffic_data)

                ret, buffer = cv2.imencode('.jpg', frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    await self.send(bytes_data=buffer.tobytes())

                frame_count += 1
                await asyncio.sleep(frame_interval)

            cap.release()

            await self.send(text_data=json.dumps({
                'stream_id':stream_id,
                'traffic_data': traffic_data
            }))


def determine_lane(bbox, lane_boundaries):
    """
    通过使用实际车道边界来完善车道确定逻辑。
    lane_boundaries（车道边界）： 标记每个车道边界的 x 轴位置列表。
    """
    # 假设车道间距相等，则根据 bbox 水平位置分配车道
    x1 = bbox[0]
    for i, boundary in enumerate(lane_boundaries):
        if x1 < boundary:
            return i
    return len(lane_boundaries)  # 对于 bbox 超过最后边界的情况


def determine_flow_index(direction):
    """
    改进了方向到流量指数的映射，使方向类别更加细化。
    - 使用 8 个心轴方向和心轴间方向。
    """
    # 将方向映射到流量矩阵中的索引
    # 定义 8 个可能的方向（北、东北、东等）
    if -np.pi / 8 <= direction < np.pi / 8:
        return (0, 1)  # East
    elif np.pi / 8 <= direction < 3 * np.pi / 8:
        return (1, 1)  # North-East
    elif 3 * np.pi / 8 <= direction < 5 * np.pi / 8:
        return (1, 0)  # North
    elif 5 * np.pi / 8 <= direction < 7 * np.pi / 8:
        return (1, -1)  # North-West
    elif -3 * np.pi / 8 <= direction < -np.pi / 8:
        return (0, -1)  # South-East
    elif -5 * np.pi / 8 <= direction < -3 * np.pi / 8:
        return (-1, -1)  # South-West
    elif -7 * np.pi / 8 <= direction < -5 * np.pi / 8:
        return (-1, 0)  # South
    else:
        return (-1, 1)  # West


def detect_abnormal_behavior(traffic_data, crowd_paths):
    for track_id, path in crowd_paths.items():
        if len(path) > 10:
            if rapid_gathering(path):
                traffic_data['incident_count'] += 1
                print("发现潜在的踩踏事件")
            if sudden_dispersal(path):
                traffic_data['incident_count'] += 1
                print("Sudden dispersal detected")


def rapid_gathering(path, min_points=5, area_threshold=10):
    """
    增强的快速聚集检测逻辑：
    - 检测路径上的点是否随着时间的推移迅速聚集到一个小区域。
    """
    # 检测路径点是否快速汇聚到一个小区域
    if len(path) < min_points:
        return False
    # 测量最早的点和最近的点之间的距离
    dist = np.linalg.norm(np.array(path[-1]) - np.array(path[0]))
    # 测量路径上最后几个点所覆盖的面积
    area = np.max(path,axis=0) - np.min(path,axis=0)
    area_coverage = np.linalg.norm(area)

    return dist < area_threshold and area_coverage < area_threshold  # 检测到快速采集


def sudden_dispersal(path, dispersal_threshold = 50):
    """
    通过检查路径上的点是否随着时间的推移而迅速扩散，检测突然的分散。
    """
    dist = np.linalg.norm(np.array(path[-1]) - np.array(path[0]))
    return dist > dispersal_threshold


def analyze_activity_by_time(current_time, traffic_data, activity_data, rush_hour_ranges):
    """
    扩展活动分析，以处理不同的阈值并检测高峰时段。
    - rush_hour_ranges： 定义高峰时段开始和结束时间的图元列表。
    """
    # 跟踪当前时间段的总移动量
    total_movement = traffic_data['people_count'] + traffic_data['vehicle_count']
    activity_data[current_time] = total_movement

    # 检查当前时间是否在高峰时段内
    for start_time, end_time in rush_hour_ranges:
        if start_time <= current_time <= end_time:
            if total_movement > 100:
                print(f"High activity detected during rush hour at {current_time}")
            return

        # 非高峰时段常规检测阈值
        if total_movement > 50:
            print(f"High activity detected at {current_time}")


def update_flow_matrix(traffic_data, traj):
    """
    改进了 update_flow_matrix 功能，支持根据方向对车辆和人员进行更细化的流量分类。
    根据方向对车辆和人员进行分类。
    """
    direction = traj['direction']
    flow_idx = determine_flow_index(direction)
    traffic_data['flow_matrix'][flow_idx[0], flow_idx[1]] += 1

    # 在视频上可视化流程（如箭头或热图）
    if 'flow_visualization' not in traffic_data:
        traffic_data['flow_visualization'] = np.zeros_like(traffic_data['flow_matrix'])
    traffic_data['flow_visualization'][flow_idx[0], flow_idx[1]] += 1


def calculate_network_capacity(traffic_data, time_frame):
    """
    根据单位时间内通过的总人数/车辆数计算道路或行人网络的容量。
    单位时间内通过的总人数。
    """
    total_vehicles = traffic_data['vehicle_count']
    total_people = traffic_data['people_count']
    capacity_per_time_frame = (total_vehicles + total_people) / time_frame
    return capacity_per_time_frame


def visualize_flow_matrix(frame, traffic_data):
    """
    将视频帧上的流量矩阵可视化为箭头或热图，显示车流和人流。
    """
    flow_matrix = traffic_data.get('flow_visualization' , np.zeros((4,4)))

    for i in range(flow_matrix.shape[0]):
        for j in range(flow_matrix.shape[1]):
            intensity = flow_matrix[i,j]
            if intensity > 0:
                start_point = (frame.shape[1] // 4 * j, frame.shape[0] // 4 * i)
                end_point = (start_point[0] + int(20 * intensity), start_point[1] + int (20 * intensity))
                cv2.arrowedLine(frame, start_point, end_point, (0,0,255), 2)

    return frame


def dynamic_congestion_detection(traffic_data, historical_density_data):
    """
    根据历史数据校准阈值，动态改进拥塞检测。
    """
    current_density = traffic_data['vehicle_density'] + traffic_data['people_density']
    historical_avg_density = np.mean(historical_density_data)

    # 如果当前密度大大超过历史平均值，就会检测到拥堵情况
    if current_density > 1.5 * historical_avg_density:
        print('检测到拥堵！')
    else:
        print("未检测到拥堵。")


# 检测速度骤降
def detect_sudden_speed_drop(traj, speed_threshold=10):
    """
    检测物体的速度是否突然下降（表明存在潜在危险）。
    speed_threshold（速度阈值）： 速度突然下降的幅度。
    """
    if len(traj['speeds']) < 5:
        return False

    # 检查速度是否比以前明显下降
    previous_speed = traj['speeds'][-2]
    current_speed = traj['speeds'][-1]
    speed_drop = previous_speed - current_speed

    return speed_drop > speed_threshold


# 检测到在非停车区域长时间停车
def detect_prolonged_halt(traj, halt_threshold=30):
    """
    检测物体是否在非停车区域停留时间过长。
    halt_threshold（停止阈值）： 物体保持静止的连续帧数。
    """
    if traj['speed'] < 0.1:  # 几乎固定
        traj['stationary_frames'] = traj.get('stationary_frames', 0) + 1
        if traj['stationary_frames'] > halt_threshold:
            return True
    else:
        traj['stationary_frames'] = 0
    return False


# 检测意外的方向变化
def detect_unexpected_direction_change(traj, direction_threshold=np.pi/4):
    """
    检测对象是否在超出预期流量的情况下突然改变方向。
    方向阈值： 被视为异常的方向角度变化。
    """
    if len(traj['directions']) < 5:
        return False  # 没有足够的数据来检测模式

    # 计算最后两个方向的角度差
    previous_direction = traj['directions'][-2]
    current_direction = traj['directions'][-1]
    angular_change = abs(previous_direction - current_direction)

    return angular_change > direction_threshold


# 用于事件检测的实例
def handle_abnormal_behaviors(traffic_data, traj, frame):
    """
    将异常行为检查纳入流量监控环路。
    """
    if detect_sudden_speed_drop(traj):
        traffic_data['incident_count'] += 1
        cv2.putText(frame, '检测到速度突然下降', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if detect_prolonged_halt(traj):
        traffic_data['incident_count'] += 1
        cv2.putText(frame, '检测到长时间停顿', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if detect_unexpected_direction_change(traj):
        traffic_data['incident_count'] += 1
        cv2.putText(frame, '意想不到的方向变化', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


# 处理事故（超速、逆向行驶、长时间停车）
def handle_incidents(traffic_data, traj, frame):
    speed_limit = 50
    if traj['speed'] > speed_limit:
        traffic_data['incident_count'] += 1
        cv2.putText(frame, '超速', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if abs(traj['direction']) > np.pi / 2:
        traffic_data['incident_count'] += 1
        cv2.putText(frame, '检测到走错路', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


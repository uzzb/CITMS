import json
import os

import cv2
import torch
import asyncio
import numpy as np
from collections import deque
from channels.generic.websocket import AsyncWebsocketConsumer
import torchvision
from natsort import natsorted

from traffic_management.services.db_services import insert_traffic_data

class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        directory = r'E:\道路视频\南三国旗杆台阶_20241015174856'
        directory = r'E:\道路视频\南三休息室后外_20241015173030'
        directory = r'E:\道路视频\南四舍与南五舍南出口26_20241015172824'

        video_files = []
        for root, dirs, files in os.walk(directory):
            for file in natsorted(files):
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

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
                'flow_matrix': np.zeros((4,4)),
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

                if frame_count % detection_interval == 0:
                    frame_tensor = torch.from_numpy(frame).float().cuda()
                    if len(frame_tensor.shape) == 3:
                        frame_tensor = frame_tensor.permute(2,0,1)
                    frame_tensor = frame_tensor.unsqueeze(0)

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            results = model(frame_tensor)

                    iou_threshold = 0.4
                    detections = results[0]
                    keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_threshold)
                    detections = detections[keep]

                people_count = 0
                vehicle_count = 0
                frame_area = frame.shape[0] * frame.shape[1]

                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection[:6].tolist()
                    label = model.names[int(cls)]
                    if label in ['person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus']:
                        center_y, center_x = int((y1 + y2) / 2), int((x1 + x2) / 2)
                        if fgmask[center_y, center_x] > 0:
                            width, height = x2 - x1, y2 - y1
                            if width > 0 and height > 0:
                                p1 = (int(x1), int(y1))
                                p2 = (int(x2), int(y2))
                                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                                cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (0, 255, 0), 2)

                                if label == 'person':
                                    people_count += 1
                                elif label in ['car', 'bus', 'truck']:
                                    vehicle_count += 1

                people_density = people_count / frame_area
                vehicle_density = vehicle_count / frame_area
                traffic_data['people_density'] = people_density
                traffic_data['vehicle_density'] = vehicle_density
                historical_density_data.append(people_density + vehicle_density)

                if people_density > 0.05:
                    cv2.putText(frame, 'Crowded Area Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if vehicle_density > 0.05:
                    cv2.putText(frame, 'Traffic Congestion Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if frame_count % (detection_interval * 30) == 0:
                    insert_traffic_data(stream_id, traffic_data)

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
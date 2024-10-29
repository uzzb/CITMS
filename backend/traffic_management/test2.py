import json
import os
import traceback
import cv2
import numpy as np
import torch
import asyncio

import torchvision
from channels.generic.websocket import AsyncWebsocketConsumer
from natsort import natsorted
from yolov5.utils.augmentations import letterbox


class VideoStreamConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task = None
        self.is_streaming = False
        self.model = None
        self._connection_open = False

    async def connect(self):
        self._connection_open = True
        self.selected_video_index = 0
        self.is_streaming = False
        if self.model is None:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            self.model = self.model.to('cuda')
            self.model.conf = 0.5
            self.model.iou = 0.45
            self.model.eval()
        await self.accept()

    async def disconnect(self, close_code):
        self._connection_open = False
        self.is_streaming = False
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)

            if 'playbackSpeed' in data:
                self.playback_speed = float(data['playbackSpeed'])
                return

            new_index = data.get('selectedVideoIndex', self.selected_video_index)
            if new_index != self.selected_video_index:
                self.is_streaming = False
                if self.current_task:
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass

                await asyncio.sleep(0.1)

                self.selected_video_index = new_index
                self.is_streaming = True

                video_directories = [
                    r'E:\道路视频\南三国旗杆台阶_20241015174856',
                    r'E:\道路视频\南三休息室后外_20241015173030',
                    r'E:\道路视频\南四舍与南五舍南出口26_20241015172824'
                ]

                self.current_task = asyncio.create_task(
                    self.stream_videos_from_directory(video_directories[self.selected_video_index])
                )

        except Exception as e:
            print(f"Error in receive: {e}")

    async def stream_videos_from_directory(self, directory):
        try:
            video_files = []
            print(f"Scanning directory: {directory}")

            for root, dirs, files in os.walk(directory):
                for file in natsorted(files):
                    if file.endswith('.mp4'):
                        video_files.append(os.path.join(root, file))

            print(f"Found {len(video_files)} video files")

            if not video_files:
                if self._connection_open:
                    await self.safe_send(text_data=json.dumps({
                        'error': f'No video files found in directory: {directory}'
                    }))
                return

            for video_file in video_files:
                if not self.is_streaming or not self._connection_open:
                    break
                print(f"Processing video file: {video_file}")
                await self.process_video(video_file)

        except Exception as e:
            print(f"Error in stream_videos_from_directory: {e}")
            if self._connection_open:
                await self.safe_send(text_data=json.dumps({
                    'error': f'Error processing directory: {str(e)}'
                }))

    async def safe_send(self, **kwargs):
        if self._connection_open and self.is_streaming:
            try:
                await self.send(**kwargs)
            except Exception as e:
                print(f"Error sending message: {e}")
                self.is_streaming = False
                return False
            return True
        return False

    async def process_frame(self, frame):
        try:
            # 不使用任何预处理，直接使用原始帧
            results = self.model(frame)

            # 获取检测结果并进行NMS处理
            conf_threshold = 0.5
            iou_threshold = 0.45
            detections = results.xyxy[0]
            keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_threshold)
            detections = detections[keep]

            # 遍历检测结果
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.tolist()
                if conf < conf_threshold:
                    continue

                cls = int(cls)
                if cls < len(self.model.names):
                    label = self.model.names[cls]

                    if label in ['person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus']:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        x2 = max(0, min(x2, frame.shape[1] - 1))
                        y2 = max(0, min(y2, frame.shape[0] - 1))

                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                            label_text = f'{label} {conf:.2f}'
                            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]

                            if y1 - text_size[1] - 2 >= 0:
                                cv2.rectangle(frame, (x1, y1 - text_size[1] - 2),
                                              (x1 + text_size[0], y1), (0, 255, 0), -1)
                                cv2.putText(frame, label_text, (x1, y1 - 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                            else:
                                cv2.rectangle(frame, (x1, y2),
                                              (x1 + text_size[0], y2 + text_size[1] + 2), (0, 255, 0), -1)
                                cv2.putText(frame, label_text, (x1, y2 + text_size[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            return frame

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame

    async def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if self._connection_open:
                await self.safe_send(text_data=json.dumps({
                    'error': f'Unable to open video: {video_path}'
                }))
            return

        try:
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = 1.0 / original_fps
            last_frame_time = asyncio.get_event_loop().time()

            while cap.isOpened() and self.is_streaming and self._connection_open:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - last_frame_time

                if elapsed >= frame_interval:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process single frame
                    processed_frame = await self.process_frame(frame)

                    # Resize for transmission while maintaining aspect ratio
                    scale_factor = 0.5
                    new_width = int(processed_frame.shape[1] * scale_factor)
                    new_height = int(processed_frame.shape[0] * scale_factor)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))

                    ret, buffer = cv2.imencode(
                        '.jpg',
                        processed_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )

                    if ret and self._connection_open and self.is_streaming:
                        await self.safe_send(bytes_data=buffer.tobytes())
                        last_frame_time = current_time

                await asyncio.sleep(max(0, frame_interval - elapsed))

        except Exception as e:
            print(f"Error processing video: {e}")
            if self._connection_open:
                await self.safe_send(text_data=json.dumps({
                    'error': f'Error processing video: {str(e)}'
                }))
        finally:
            cap.release()
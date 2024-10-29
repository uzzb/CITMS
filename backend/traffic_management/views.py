import json
import os
import traceback
import cv2
import numpy as np
import torch
import asyncio

import torchvision
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.http import JsonResponse
from django.db import transaction
from natsort import natsorted
from yolov5.utils.augmentations import letterbox

from traffic_management.models import Location, AnalysisRecord
from traffic_management.services.enhanced_frame_processor import EnhancedFrameProcessor, AnalysisData
from traffic_management.services.frame_processor import FrameProcessor


class VideoStreamConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task = None
        self.is_streaming = False
        self.model = None
        self._connection_open = False
        self.frame_processors = {}  # 处理器
        self.analysis_interval = 1.0
        self.last_analysis_time = 0

        self.locations = {
            0: "南三国旗杆台阶",
            1: "南三休息室后",
            2: "南四舍与南五舍南出口"
        }

    @database_sync_to_async
    def get_or_create_location(self, location_id, name):
        """同步数据库操作包装为异步"""
        with transaction.atomic():
            location, created = Location.objects.get_or_create(
                id=location_id + 1,  # 数据库ID从1开始
                defaults={'name': name}
            )
            return location, created


    async def connect(self):
        self._connection_open = True
        self.selected_video_index = 0
        self.is_streaming = False

        # 初始化模型
        if self.model is None:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            self.model = self.model.to('cuda')
            self.model.conf = 0.5
            self.model.iou = 0.45
            self.model.eval()

        # 使用 self.locations 来初始化位置和处理器
        for location_id, name in self.locations.items():
            # 确保数据库中存在该位置
            location, created = await self.get_or_create_location(location_id, name)

            # 创建处理器实例
            self.frame_processors[location_id] = EnhancedFrameProcessor(
                location_id=location.id,
                location_name=name
            )

        # 设置默认处理器
        self.frame_processor = self.frame_processors[0]
        await self.accept()

    @database_sync_to_async
    def get_location_statistics(self, location_id):
        """包装同步的统计数据获取操作"""
        return AnalysisRecord.get_location_statistics(location_id)

    async def send_analysis_data(self, analysis_data: AnalysisData):
        """发送分析数据到客户端"""
        if not self._connection_open or not self.is_streaming:
            return

        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return

        try:
            # 获取历史统计数据
            if analysis_data.location_id:
                stats = await self.get_location_statistics(analysis_data.location_id)
            else:
                stats = {}

            # 转换数据为JSON格式
            data = {
                'type': 'analysis',
                'data': {
                    **analysis_data.to_dict(),
                    'statistics': stats
                }
            }

            await self.safe_send(text_data=json.dumps(data))
            self.last_analysis_time = current_time

        except Exception as e:
            print(f"Error sending analysis data: {e}")

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
                self.frame_processor = self.frame_processors[new_index]
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

                    # 处理帧并获取分析数据
                    processed_frame, analysis_data = await self.frame_processor.process_frame(frame, self.model)

                    # 发送分析数据
                    if analysis_data:
                        await self.send_analysis_data(analysis_data)

                    # 调整尺寸并发送视频帧
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

    def get_location_analysis(request, location_id):
        stats = AnalysisRecord.get_location_statistics(location_id)
        return JsonResponse(stats)
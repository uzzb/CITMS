import traceback

import cv2
import numpy as np
import torch
import torchvision
from typing import Tuple


class FrameProcessor:
    VALID_LABELS = {'person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus'}
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.3
    FONT_THICKNESS = 1
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 0, 0)
    BOX_THICKNESS = 1
    PADDING = 2

    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @staticmethod
    def _clip_coordinates(x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, ...]) -> Tuple[
        int, int, int, int]:
        height, width = frame_shape[:2]
        return (
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(0, min(x2, width - 1)),
            max(0, min(y2, height - 1))
        )

    def _draw_label(self, frame: np.ndarray, label_text: str, x1: int, y1: int, y2: int) -> None:
        text_size = cv2.getTextSize(label_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)[0]

        if y1 - text_size[1] - self.PADDING >= 0:
            text_y = y1 - self.PADDING
            bg_y1 = y1 - text_size[1] - self.PADDING
            bg_y2 = y1
        else:
            text_y = y2 + text_size[1]
            bg_y1 = y2
            bg_y2 = y2 + text_size[1] + self.PADDING

        cv2.rectangle(frame, (x1, bg_y1), (x1 + text_size[0], bg_y2), self.BOX_COLOR, -1)
        cv2.putText(frame, label_text, (x1, text_y), self.FONT, self.FONT_SCALE,
                    self.TEXT_COLOR, self.FONT_THICKNESS)

    async def process_frame(self, frame: np.ndarray, model: torch.nn.Module) -> np.ndarray:
        try:
            with torch.no_grad():
                results = model(frame)

            detections = results.xyxy[0]
            keep = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                self.iou_threshold
            )
            detections = detections[keep]

            mask = detections[:, 4] >= self.conf_threshold
            detections = detections[mask]

            for detection in detections:
                x1, y1, x2, y2, conf, cls = map(float, detection.tolist())
                cls = int(cls)

                if cls >= len(model.names):
                    continue

                label = model.names[cls]
                if label not in self.VALID_LABELS:
                    continue

                x1, y1, x2, y2 = self._clip_coordinates(
                    int(x1), int(y1), int(x2), int(y2), frame.shape
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, self.BOX_THICKNESS)

                label_text = f'{label} {conf:.2f}'
                self._draw_label(frame, label_text, x1, y1, y2)

            return frame

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame


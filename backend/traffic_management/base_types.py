# traffic_management/base_types.py
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Any
from datetime import datetime
import numpy as np

@dataclass
class AnalysisData:
    """分析数据结构"""
    zone_counts: Dict[str, int]
    warnings: Set[str]
    crowd_density: float
    total_count: int
    velocity: float
    abnormal_events: List[str]
    location_id: int = None
    timestamp: datetime = None

    zone_densities: Dict[str, float] = field(default_factory=dict)  # 每个区域的密度
    acceleration: float = 0.0  # 平均加速度
    path_predictions: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # 路径预测

    traffic_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """转换为可序列化的字典"""
        return {
            'location_id': self.location_id,
            'zone_counts': self.zone_counts,
            'warnings': list(self.warnings),
            'crowd_density': float(self.crowd_density),
            'total_count': self.total_count,
            'velocity': float(self.velocity),
            'abnormal_events': self.abnormal_events,
            'timestamp': self.timestamp.isoformat(),
            'traffic_analysis': self.traffic_analysis,
        }

# 预定义区域配置
LOCATION_CONFIGS = {
    "南三国旗杆台阶": {
        "zones": {
            "台阶区": np.array([
                [150, 250],  # 左上
                [450, 250],  # 右上
                [500, 400],  # 右下
                [100, 400]   # 左下
            ]),
            "人行道": np.array([
                [100, 100],  # 左上
                [500, 100],  # 右上
                [500, 250],  # 右下
                [100, 250]   # 左下
            ]),
            "绿化带": np.array([
                [500, 100],  # 左上
                [600, 100],  # 右上
                [600, 400],  # 右下
                [500, 400]   # 左下
            ])
        },
        "thresholds": {
            "crowd": 0.5,    # 台阶区域需要更低的拥挤阈值
            "velocity": 1.2,  # 台阶区域速度阈值更低
            "density": 0.4,   # 保持较低的密度阈值
            "acceleration": 0.6,  # 降低加速度阈值，因为是台阶区域
            "zone_specific": {
                "台阶区": {
                    "crowd": 0.4,     # 台阶区更严格的拥挤控制
                    "velocity": 1.0,   # 台阶区更低的速度限制
                    "direction_flow": True  # 启用方向流动监控
                },
                "人行道": {
                    "crowd": 0.6,     # 人行道可以有较高的人流量
                    "velocity": 1.5    # 人行道允许较快的行走速度
                }
            }
        }
    },
    "南三休息室后": {
        "zones": {
            "主通道": np.array([
                [150, 100],   # 左上角
                [350, 100],   # 右上角
                [350, 300],   # 右下角
                [150, 300]    # 左下角
            ]),
            "绿化带": np.array([
                [50, 100],    # 左上角
                [150, 100],   # 右上角
                [150, 300],   # 右下角
                [50, 300]     # 左下角
            ]),
            "树木区": np.array([
                [350, 100],   # 左上角
                [400, 100],   # 右上角
                [400, 300],   # 右下角
                [350, 300]    # 左下角
            ])
        },
        "thresholds": {
            "crowd": 0.5,      # 降低拥挤阈值，因为是休息区域
            "velocity": 1.5,   # 降低速度阈值，因为是人行道
            "density": 0.4     # 调整密度阈值以适应实际场景
        },
        "attributes": {
            "type": "outdoor",
            "lighting": "natural",
            "surface": "paved_path",
            "vegetation": True
        }
    },
    "南四舍与南五舍南出口": {
        "zones": {
            "出口区": np.array([
                [200, 200],  # 左上，靠近建筑物出口
                [400, 200],  # 右上
                [400, 400],  # 右下
                [200, 400]   # 左下
            ]),
            "等候区": np.array([
                [50, 200],   # 左上，靠墙休息区域
                [200, 200],  # 右上
                [200, 400],  # 右下
                [50, 400]    # 左下
            ]),
            "主干道": np.array([
                [400, 150],  # 左上
                [600, 150],  # 右上
                [600, 400],  # 右下
                [400, 400]   # 左下
            ])
        },
        "thresholds": {
            "crowd": 0.7,    # 降低拥挤阈值，考虑到是出入口
            "velocity": 1.8,  # 降低速度阈值，因为有人群聚集
            "density": 0.5,   # 适应出入口人流密度
            "zone_specific": {
                "等候区": {
                    "crowd": 0.8,     # 等候区可以容许更高的人群密度
                    "velocity": 1.0    # 等候区速度阈值较低
                }
            }
        }
    },
}
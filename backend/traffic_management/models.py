# traffic_management/models.py
from django.db import models
import json


class Location(models.Model):
    """监控位置模型"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'locations'


class AnalysisRecord(models.Model):
    """分析数据记录模型"""
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    zone_counts = models.TextField()
    warnings = models.TextField()
    crowd_density = models.FloatField()
    total_count = models.IntegerField()
    velocity = models.FloatField()
    abnormal_events = models.TextField()

    max_zone_density = models.FloatField(default=0.0)  # 记录区域最大密度
    acceleration = models.FloatField(default=0.0)  # 记录平均加速度
    peak_hours = models.TextField(default='[]')  # 记录高峰期时间

    class Meta:
        db_table = 'analysis_records'
        ordering = ['-timestamp']

    def set_zone_counts(self, data):
        self.zone_counts = json.dumps(data)

    def get_zone_counts(self):
        return json.loads(self.zone_counts)

    def set_warnings(self, data):
        self.warnings = json.dumps(list(data))

    def get_warnings(self):
        return set(json.loads(self.warnings))

    def set_abnormal_events(self, data):
        self.abnormal_events = json.dumps(data)

    def get_abnormal_events(self):
        return json.loads(self.abnormal_events)

    @classmethod
    def save_analysis(cls, location_id: int, analysis_data):
        record = cls(
            location_id=location_id,
            crowd_density=analysis_data.crowd_density,
            total_count=analysis_data.total_count,
            velocity=analysis_data.velocity,
            max_zone_density=max(analysis_data.zone_densities.values(), default=0.0),
            acceleration=analysis_data.acceleration
        )
        record.set_zone_counts(analysis_data.zone_counts)
        record.set_warnings(analysis_data.warnings)
        record.set_abnormal_events(analysis_data.abnormal_events)
        record.save()
        return record

    @classmethod
    def get_location_statistics(cls, location_id: int) -> dict:
        """获取位置统计信息"""
        from django.db.models import Avg, Max, Count
        from django.db.models.functions import TruncDate

        stats = cls.objects.filter(location_id=location_id).aggregate(
            avg_density=Avg('crowd_density'),
            avg_count=Avg('total_count'),
            max_count=Max('total_count'),
            days_recorded=Count(TruncDate('timestamp'), distinct=True)
        )
        return stats


class TrafficAnalysisRecord(models.Model):
    """交通分析记录模型"""
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    # 路段分析
    lane_occupancy = models.TextField(default=dict)  # 车道占用率
    traffic_capacity = models.TextField(default=dict)  # 通行能力
    saturation = models.FloatField(default=0.0)  # 饱和度

    # 流向分析
    flow_matrix = models.TextField(default=dict)  # 流向矩阵
    main_directions = models.TextField(default=list)  # 主要流向

    # 速度分析
    avg_speed = models.FloatField(default=0.0)
    speed_distribution = models.TextField(default=dict)  # 速度分布
    passage_times = models.TextField(default=dict)  # 通过时间

    # 异常事件
    bottleneck_points = models.TextField(default=list)  # 瓶颈点
    congestion_areas = models.TextField(default=list)  # 拥堵区域
    abnormal_events = models.TextField(default=list)  # 异常事件

    class Meta:
        db_table = 'traffic_analysis_records'
        ordering = ['-timestamp']

    @classmethod
    def save_analysis(cls, location_id: int, traffic_data: dict):
        """保存交通分析数据"""
        record = cls(
            location_id=location_id,
            lane_occupancy=traffic_data.get('lane_occupancy', {}),
            traffic_capacity=traffic_data.get('traffic_capacity', {}),
            saturation=traffic_data.get('saturation', 0.0),
            flow_matrix=traffic_data.get('flow_matrix', {}),
            main_directions=traffic_data.get('main_directions', []),
            avg_speed=traffic_data.get('avg_speed', 0.0),
            speed_distribution=traffic_data.get('speed_distribution', {}),
            passage_times=traffic_data.get('passage_times', {}),
            bottleneck_points=traffic_data.get('bottleneck_points', []),
            congestion_areas=traffic_data.get('congestion_areas', []),
            abnormal_events=traffic_data.get('abnormal_events', [])
        )
        record.save()
        return record
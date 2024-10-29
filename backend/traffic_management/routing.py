from django.urls import re_path
from . import views


'''
    Django Channels的一部分，它拓展了Django以处理WebSockets、HTTP2等
    为ws/video_stream/设置了一个WebSocket路由，任何指向该URL的连接都将由VideoStreamConsumer处理
    re_path: Django函数，用于使用正则表达式定义URL路由
    r'ws/video_stream/$ ws表示WebSocket连接，video_stream/表示特定端点
    view.VideoStreamConsumer.as_asgi()调用VideoStreamConsumer类的as_asgi()方法，将其转化为兼容ASGI的可调用程序
'''
websocket_urlpatterns = [
    re_path(r'ws/video_stream/$', views.VideoStreamConsumer.as_asgi()),
]

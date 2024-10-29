import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketVideoStreamWidget extends StatefulWidget {
  final Function(WebSocketVideoStreamWidgetState)? onWebSocketInitialized;

  final Function(Map<String, dynamic>)? onAnalysisDataReceived;

  const WebSocketVideoStreamWidget({
    Key? key,
    this.onWebSocketInitialized,
    this.onAnalysisDataReceived,  // 新添加的回调
  }) : super(key: key);

  @override
  WebSocketVideoStreamWidgetState createState() => WebSocketVideoStreamWidgetState();
}

class WebSocketVideoStreamWidgetState extends State<WebSocketVideoStreamWidget> {
  WebSocketChannel? _channel;
  Uint8List? _imageBytes;
  bool _isProcessingIndex = false;
  bool _isLoading = true;
  bool _isConnecting = false;
  bool _hasReceivedFirstFrame = false;
  Timer? _reconnectionTimer;
  int _reconnectAttempts = 0;
  static const int maxReconnectAttempts = 5;
  Timer? _heartbeatTimer;
  Completer<void>? _wsConnectedCompleter;

  void _startHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      if (_channel != null) {
        try {
          _channel!.sink.add(json.encode({'type': 'heartbeat'}));
        } catch (e) {
          print('Error sending heartbeat: $e');
          _handleConnectionError();
        }
      }
    });
  }


  void _handleError(String error) {
    print('Error from server: $error');
    setState(() {
      _isLoading = false;
    });
  }

  void _handleConnectionError() {
    _isConnecting = false;
    if (_reconnectAttempts < maxReconnectAttempts) {
      _scheduleReconnection();
    } else {
      print('Max reconnection attempts reached');
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _handleConnectionClosed() {
    _isConnecting = false;
    _heartbeatTimer?.cancel();
    if (_reconnectAttempts < maxReconnectAttempts) {
      _scheduleReconnection();
    }
  }

  void _handleWebSocketMessage(dynamic data) {
    if (data is List<int>) {
      // 处理视频帧
      setState(() {
        _imageBytes = Uint8List.fromList(data);
        _hasReceivedFirstFrame = true;
        _isLoading = false;
      });
    } else if (data is String) {
      try {
        final jsonData = json.decode(data);
        if (jsonData['type'] == 'analysis') {
          // 调用分析数据回调
          widget.onAnalysisDataReceived?.call(jsonData['data']);
        } else if (jsonData['error'] != null) {
          print('Server error: ${jsonData['error']}');
          _handleError(jsonData['error']);
        }
      } catch (e) {
        print('Error parsing message: $e');
      }
    }
  }

  Future<bool> initWebSocket() async {
    if (_isConnecting) return false;
    _isConnecting = true;
    _wsConnectedCompleter = Completer<void>();

    try {
      print('Attempting to connect to WebSocket...');
      const String serverUrl = "ws://192.168.42.120:8000/ws/video_stream/";

      _channel = IOWebSocketChannel.connect(
        serverUrl,
        pingInterval: const Duration(seconds: 5),
      );

      _channel!.stream.listen(
        _handleWebSocketMessage,
        onError: (error) {
          print('WebSocket error: $error');
          _handleConnectionError();
          _wsConnectedCompleter?.completeError(error);
        },
        onDone: () {
          print('WebSocket connection closed');
          _handleConnectionClosed();
        },
        cancelOnError: false,
      );


      await _channel!.ready;

      _startHeartbeat();
      _isConnecting = false;
      _reconnectAttempts = 0;
      print('WebSocket connected successfully');
      _wsConnectedCompleter?.complete();
      return true;

    } catch (e) {
      print('WebSocket connection failed: $e');
      _handleConnectionError();
      _wsConnectedCompleter?.completeError(e);
      return false;
    }
  }


  void _scheduleReconnection() {
    _isConnecting = false;
    _reconnectionTimer?.cancel();
    _heartbeatTimer?.cancel();

    final backoffDuration = Duration(seconds: _calculateBackoffTime());
    _reconnectionTimer = Timer(backoffDuration, () {
      print('Attempting to reconnect... (Attempt ${_reconnectAttempts + 1})');
      _reconnectAttempts++;
      initWebSocket();
    });
  }



  int _calculateBackoffTime() {
    return (2 * (_reconnectAttempts + 1)).clamp(2, 30);
  }

  @override
  void initState() {
    super.initState();
    widget.onWebSocketInitialized?.call(this);

    if (Platform.isAndroid || Platform.isIOS) {
      _initializeVideoStream();
    } else {
      print('Unsupported platform');
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _initializeVideoStream() async {
    try {
      final connected = await initWebSocket();
      if (connected) {
        await _wsConnectedCompleter?.future;
        await Future.delayed(const Duration(milliseconds: 500));
        sendVideoIndex(0);
      }
    } catch (e) {
      print('Error initializing video stream: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> sendVideoIndex(int index) async {
    if (_isProcessingIndex) return;

    try {
      _isProcessingIndex = true;
      setState(() {
        _imageBytes = null;
        _isLoading = true;
        _hasReceivedFirstFrame = false;
      });

      if (_channel != null && _channel!.sink != null) {
        print('Switching to video index: $index');
        _channel!.sink.add(json.encode({'selectedVideoIndex': index}));
        print('Video index sent successfully');
      } else {
        print('WebSocket not connected');
        await initWebSocket();
      }
    } catch (e) {
      print('Error sending video index: $e');
      setState(() {
        _isLoading = false;
      });
    } finally {
      _isProcessingIndex = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: _imageBytes != null
          ? Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width,
          maxHeight: MediaQuery.of(context).size.height,
        ),
        child: Image.memory(
          _imageBytes!,
          gaplessPlayback: true,
          fit: BoxFit.contain,
          filterQuality: FilterQuality.medium,
        ),
      )
          : _isLoading
          ? const SizedBox(
        width: 60,
        height: 60,
        child: CircularProgressIndicator(),
      )
          : const Icon(
        Icons.error_outline,
        size: 60,
        color: Colors.red,
      ),
    );
  }

  @override
  void dispose() {
    _reconnectionTimer?.cancel();
    _heartbeatTimer?.cancel();
    _channel?.sink.close();
    super.dispose();
  }
}
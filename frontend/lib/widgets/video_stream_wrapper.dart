import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:front/widgets/web_socket_video_stream_widget.dart';

class VideoStreamWrapper extends StatefulWidget {
  final GlobalKey<WebSocketVideoStreamWidgetState> webSocketKey;
  final Function(dynamic) onAnalysisDataReceived;

  const VideoStreamWrapper({
    Key? key,
    required this.webSocketKey,
    required this.onAnalysisDataReceived,
  }) : super(key: key);

  @override
  State<VideoStreamWrapper> createState() => _VideoStreamWrapperState();
}

class _VideoStreamWrapperState extends State<VideoStreamWrapper>
    with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);

    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey[300]!),
        borderRadius: BorderRadius.circular(8),
      ),
      child: WebSocketVideoStreamWidget(
        key: widget.webSocketKey,
        onWebSocketInitialized: (state) {},
        onAnalysisDataReceived: widget.onAnalysisDataReceived,
      ),
    );
  }
}
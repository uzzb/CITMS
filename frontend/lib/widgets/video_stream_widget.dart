import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:front/services/api_service.dart';

class VideoStreamWidget extends StatefulWidget {
  @override
  _VideoStreamWidgetState createState() => _VideoStreamWidgetState();
}

class _VideoStreamWidgetState extends State<VideoStreamWidget> {
  Uint8List? _imageBytes;
  late StreamSubscription<List<int>> _streamSubscription;
  final StringBuffer _buffer = StringBuffer();

  @override
  void initState() {
    super.initState();
    _startVideoStream();
  }

  void _startVideoStream() async {
    final apiService = ApiService();
    final videoUrl = "E:/ProgramProject/PythonProject/Video/video.mp4";

    try {

      Stream<List<int>> videoStream = apiService.sendVideoUrlAndReceiveVideo(videoUrl);


      _streamSubscription = videoStream.listen(
            (data) {

          _buffer.write(String.fromCharCodes(data));
          _parseMjpegStream();
        },
        onError: (error) {
          print('Stream error: $error');
        },
        onDone: () {
          print('Stream done');
        },
        cancelOnError: true,
      );
    } catch (e) {
      print('Failed to stream video: $e');
    }
  }


  void _parseMjpegStream() {
    final boundary = '--frame';
    String bufferString = _buffer.toString();


    while (bufferString.contains(boundary)) {
      int startIdx = bufferString.indexOf(boundary);
      int endIdx = bufferString.indexOf(boundary, startIdx + boundary.length);


      if (endIdx != -1) {
        String frameData = bufferString.substring(startIdx + boundary.length, endIdx);


        final imageBytes = Uint8List.fromList(frameData.codeUnits);
        setState(() {
          _imageBytes = imageBytes;
        });


        _buffer.clear();
        _buffer.write(bufferString.substring(endIdx));
        bufferString = _buffer.toString();
      } else {
        break;
      }
    }
  }

  @override
  void dispose() {
    _streamSubscription.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: _imageBytes != null
            ? Image.memory(_imageBytes!)
            : CircularProgressIndicator(),
      ),
    );
  }
}

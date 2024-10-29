import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class VideoAnalyticsWidget extends StatefulWidget {
  @override
  _VideoAnalyticsWidgetState createState() => _VideoAnalyticsWidgetState();
}

class _VideoAnalyticsWidgetState extends State<VideoAnalyticsWidget> {
  Map<String, dynamic>? _analyticsData;

  void updateAnalytics(Map<String, dynamic> data) {
    setState(() {
      _analyticsData = data;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_analyticsData == null) {
      return const SizedBox();
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('总人数: ${_analyticsData!['total_count']}'),
            Text('人群密度: ${_analyticsData!['crowd_density'].toStringAsFixed(2)}'),
            Text('平均速度: ${_analyticsData!['velocity'].toStringAsFixed(2)}'),
            if (_analyticsData!['warnings'].isNotEmpty)
              Text('警告: ${_analyticsData!['warnings'].join(', ')}'),
            if (_analyticsData!['statistics'] != null) ...[
              const Divider(),
              Text('统计数据:'),
              Text('平均人数: ${_analyticsData!['statistics']['average_count']?.toStringAsFixed(2)}'),
              Text('平均密度: ${_analyticsData!['statistics']['average_density']?.toStringAsFixed(2)}'),
              Text('平均速度: ${_analyticsData!['statistics']['average_velocity']?.toStringAsFixed(2)}'),
            ],
          ],
        ),
      ),
    );
  }
}
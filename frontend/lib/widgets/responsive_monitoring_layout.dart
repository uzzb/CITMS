import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class ResponsiveMonitoringLayout extends StatelessWidget {
  final Widget videoStream;
  final Widget analysisPanel;

  const ResponsiveMonitoringLayout({
    Key? key,
    required this.videoStream,
    required this.analysisPanel,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Column(
        children: [
          TabBar(
            tabs: const [
              Tab(text: '监控画面'),
              Tab(text: '数据分析'),
            ],
            labelStyle: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          Expanded(
            child: TabBarView(
              // 使用 PageStorage 来保持状态
              children: [
                PageStorage(
                  bucket: PageStorageBucket(),
                  child: videoStream,
                ),
                PageStorage(
                  bucket: PageStorageBucket(),
                  child: SingleChildScrollView(child: analysisPanel),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
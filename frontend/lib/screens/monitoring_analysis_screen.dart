import 'dart:math';

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../util/DirectoryPainter.dart';
import '../util/HeatmapPainter.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/responsive_monitoring_layout.dart';
import '../widgets/video_stream_wrapper.dart';
import '../widgets/web_socket_video_stream_widget.dart';

class MonitoringAnalysisScreen extends StatefulWidget {
  const MonitoringAnalysisScreen({Key? key}) : super(key: key);

  @override
  _MonitoringAnalysisScreenState createState() => _MonitoringAnalysisScreenState();
}

class _MonitoringAnalysisScreenState extends State<MonitoringAnalysisScreen> {
  final GlobalKey<WebSocketVideoStreamWidgetState> _webSocketKey = GlobalKey();
  Map<String, dynamic>? _analysisData;
  int _selectedVideoIndex = 0;
  bool _showTrafficAnalysis = false;

  final List<Map<String, String>> locations = [
    {'id': '1', 'name': '南三国旗杆台阶'},
    {'id': '2', 'name': '南三休息室后'},
    {'id': '3', 'name': '南四舍与南五舍南出口'},
  ];

  void _onNavItemSelected(int index) {
    switch (index) {
      case 1:
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/user_settings');
        break;
    }
  }

  void _onTabSelected(int index) {
    setState(() {
      _selectedVideoIndex = index;
    });
    if (_webSocketKey.currentState != null) {
      _webSocketKey.currentState!.sendVideoIndex(index);
    } else {
      print("WebSocket connection not established");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        selectedIndex: 1,
        onItemSelected: _onNavItemSelected,
        title: "监控分析",
      ),
      body: Column(
        children: [
          _buildLocationSelector(),
          _buildAnalysisTypeToggle(),  // Add analysis type toggle
          Expanded(
            child: ResponsiveMonitoringLayout(
              videoStream: VideoStreamWrapper(
                webSocketKey: _webSocketKey,
                onAnalysisDataReceived: (data) {
                  setState(() {
                    _analysisData = data;
                  });
                },
              ),
              analysisPanel: _showTrafficAnalysis
                  ? _buildTrafficAnalysisPanel()
                  : _buildAnalysisPanel(),
            ),
          ),
        ],
      ),
    );
  }

  // region 车
  Widget _buildAnalysisTypeToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          Expanded(
            child: SegmentedButton<bool>(
              segments: const [
                ButtonSegment<bool>(
                  value: false,
                  label: Text('人流分析'),
                  icon: Icon(Icons.people),
                ),
                ButtonSegment<bool>(
                  value: true,
                  label: Text('交通分析'),
                  icon: Icon(Icons.traffic),
                ),
              ],
              selected: {_showTrafficAnalysis},
              onSelectionChanged: (Set<bool> newSelection) {
                setState(() {
                  _showTrafficAnalysis = newSelection.first;
                });
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrafficAnalysisPanel() {
    return Container(
      margin: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildOccupancyChart(),
            const SizedBox(height: 16),
            _buildFlowDirectionChart(),
            const SizedBox(height: 16),
            _buildSpeedDistribution(),
            const SizedBox(height: 16),
            _buildBottlenecksList(),
          ],
        ),
      ),
    );
  }

  Widget _buildOccupancyChart() {
    final occupancyData = _analysisData?['traffic_analysis']?['lane_occupancy'] as Map? ?? {};

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '车道占用率',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: BarChart(
                BarChartData(
                  barGroups: occupancyData.entries.map((e) => BarChartGroupData(
                    x: int.parse(e.key.replaceAll('lane', '')), // 修改这里来正确解析lane ID
                    barRods: [
                      BarChartRodData(
                        toY: (e.value['occupancy_rate'] as num).toDouble() * 100,
                        color: Colors.blue,
                      ),
                    ],
                  )).toList(),
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          return Text('车道${value.toInt()}');
                        },
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSpeedDistribution() {
    final speedData = _analysisData?['traffic_analysis']?['speed_stats']?['histogram'] as Map? ?? {};

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '速度分布',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  lineBarsData: [
                    LineChartBarData(
                      spots: speedData.entries.map((e) {
                        // 从范围字符串中提取中间值
                        final range = e.key.split('-');
                        final x = (double.parse(range[0]) + double.parse(range[1])) / 2;
                        return FlSpot(x, (e.value as num).toDouble());
                      }).toList(),
                      isCurved: true,
                      color: Colors.green,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFlowDirectionChart() {
    final directions = _analysisData?['traffic_analysis']?['main_directions'] as List? ?? [];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '主要流向',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: CustomPaint(
                painter: DirectionPainter(directions: directions),
                size: const Size(200, 200),
              ),
            ),
            ...directions.map((direction) => Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                '方向: ${direction['direction']} (${(direction['count'] as num).toString()}人)',
                style: const TextStyle(fontSize: 14),
              ),
            )).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildBottlenecksList() {
    final bottlenecks = _analysisData?['traffic_analysis']?['bottlenecks'] as List? ?? [];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '交通瓶颈点',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 200),
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: bottlenecks.length,
                itemBuilder: (context, index) {
                  final point = bottlenecks[index];
                  return ListTile(
                    leading: const Icon(Icons.warning, color: Colors.orange),
                    title: Text('位置: (${point['position'][0].toStringAsFixed(1)}, ${point['position'][1].toStringAsFixed(1)})'),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('拥堵程度: ${(point['severity'] * 100).toStringAsFixed(2)}%'),
                        Text('速度: ${point['velocity'].toStringAsFixed(2)} m/s'),
                      ],
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

//   endregion


  // region 人流分析
  Widget _buildAnalysisPanel() {
    return Container(
      margin: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildAnalyticsCard(),
            const SizedBox(height: 16),
            _buildDensityTrendChart(),
            const SizedBox(height: 16),
            _buildVelocityDistributionChart(),
            const SizedBox(height: 16),
            _buildCrowdHeatmap(),
            const SizedBox(height: 16),
            _buildEventTimeline(),
            const SizedBox(height: 16),
            _buildStatisticsCard(),
            const SizedBox(height: 16),
            _buildWarningsPanel(),
          ],
        ),
      ),
    );
  }

// 速度分布图表
  Widget _buildVelocityDistributionChart() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '人群速度分布',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: BarChart(
                BarChartData(
                  barGroups: [
                    BarChartGroupData(
                      x: 0,
                      barRods: [
                        BarChartRodData(
                          toY: _analysisData?['traffic_analysis']?['speed_stats']?['speed_classes']?['slow']?.toDouble() ?? 0,
                          color: Colors.green,
                        ),
                      ],
                    ),
                    BarChartGroupData(
                      x: 1,
                      barRods: [
                        BarChartRodData(
                          toY: _analysisData?['traffic_analysis']?['speed_stats']?['speed_classes']?['medium']?.toDouble() ?? 0,
                          color: Colors.yellow,
                        ),
                      ],
                    ),
                    BarChartGroupData(
                      x: 2,
                      barRods: [
                        BarChartRodData(
                          toY: _analysisData?['traffic_analysis']?['speed_stats']?['speed_classes']?['fast']?.toDouble() ?? 0,
                          color: Colors.red,
                        ),
                      ],
                    ),
                  ],
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          switch (value.toInt()) {
                            case 0:
                              return const Text('慢速');
                            case 1:
                              return const Text('中速');
                            case 2:
                              return const Text('快速');
                            default:
                              return const Text('');
                          }
                        },
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

// 统计数据卡片
  Widget _buildStatisticsCard() {
    final statistics = _analysisData?['statistics'] as Map<String, dynamic>? ?? {};

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '历史统计',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatItem(
                  '平均人数',
                  '${statistics['avg_count']?.toStringAsFixed(1) ?? '0'}',
                  Icons.groups,
                ),
                _buildStatItem(
                  '最大人数',
                  '${statistics['max_count']?.toString() ?? '0'}',
                  Icons.people_alt,
                ),
                _buildStatItem(
                  '平均密度',
                  '${(statistics['avg_density'] ?? 0).toStringAsFixed(4)}',
                  Icons.density_medium,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, IconData icon) {
    return Column(
      children: [
        Icon(icon, size: 24, color: Colors.blue),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(fontSize: 14),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

// 事件时间线
  Widget _buildEventTimeline() {
    final events = _analysisData?['abnormal_events'] as List? ?? [];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '事件时间线',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            if (events.isEmpty)
              const Center(
                child: Text('暂无异常事件'),
              )
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: events.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    leading: const Icon(Icons.event_note),
                    title: Text(events[index].toString()),
                    subtitle: Text(_analysisData?['timestamp'] ?? ''),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

// 人群密度热力图
  Widget _buildCrowdHeatmap() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '人群密度热力图',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            AspectRatio(
              aspectRatio: 16 / 9,
              child: CustomPaint(
                painter: HeatmapPainter(
                  density: _analysisData?['crowd_density'] ?? 0.0,
                  positions: _getPositionsFromData(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDensityTrendChart() {
    // 使用当前数据点
    final currentDensity = _analysisData?['crowd_density'] ?? 0.0;

    // 创建数据点
    final List<FlSpot> densitySpots = [
      FlSpot(0, currentDensity.toDouble()),
      // 如果当前值很小，添加一些测试点来验证图表是否正常工作
      FlSpot(1, 0.2),
      FlSpot(2, 0.3),
      FlSpot(3, 0.25),
      FlSpot(4, currentDensity.toDouble()),
    ];

    return SizedBox(
      height: 300,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '人群密度趋势',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  // 添加当前值显示
                  Text(
                    '当前: ${(currentDensity * 100).toStringAsFixed(2)}%',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Expanded(
                child: LineChart(
                  LineChartData(
                    gridData: FlGridData(
                      show: true,
                      drawVerticalLine: true,
                      horizontalInterval: 0.1,
                      verticalInterval: 1,
                      getDrawingHorizontalLine: (value) {
                        return FlLine(
                          color: Colors.grey.withOpacity(0.2),
                          strokeWidth: 1,
                        );
                      },
                      getDrawingVerticalLine: (value) {
                        return FlLine(
                          color: Colors.grey.withOpacity(0.2),
                          strokeWidth: 1,
                        );
                      },
                    ),
                    titlesData: FlTitlesData(
                      show: true,
                      rightTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      bottomTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 30,
                          interval: 1,
                          getTitlesWidget: (value, meta) {
                            return Text(
                              '${value.toInt()}分钟前',
                              style: const TextStyle(fontSize: 12),
                            );
                          },
                        ),
                      ),
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          interval: 0.1,
                          getTitlesWidget: (value, meta) {
                            return Padding(
                              padding: const EdgeInsets.only(right: 4),
                              child: Text(
                                '${(value * 100).toStringAsFixed(1)}%',
                                style: const TextStyle(fontSize: 12),
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                    borderData: FlBorderData(
                      show: true,
                      border: Border.all(color: Colors.grey.withOpacity(0.3)),
                    ),
                    minX: 0,
                    maxX: 4,  // 根据实际数据点数量调整
                    minY: 0,
                    maxY: max(0.4, currentDensity + 0.1),  // 动态调整Y轴范围
                    lineBarsData: [
                      LineChartBarData(
                        spots: densitySpots,
                        isCurved: true,
                        color: Colors.blue,
                        barWidth: 3,
                        isStrokeCapRound: true,
                        dotData: const FlDotData(show: true),
                        belowBarData: BarAreaData(
                          show: true,
                          color: Colors.blue.withOpacity(0.1),
                        ),
                      ),
                    ],
                    lineTouchData: LineTouchData(
                      touchTooltipData: LineTouchTooltipData(
                        tooltipBgColor: Colors.blueGrey.withOpacity(0.8),
                        tooltipRoundedRadius: 8,
                        getTooltipItems: (touchedSpots) {
                          return touchedSpots.map((LineBarSpot touchedSpot) {
                            return LineTooltipItem(
                              '密度: ${(touchedSpot.y * 100).toStringAsFixed(1)}%',
                              const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            );
                          }).toList();
                        },
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildWarningsPanel() {
    final warnings = _analysisData?['warnings'] as List? ?? [];

    return Card(
      color: warnings.isEmpty ? null : Colors.red[50],
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.warning,
                  color: warnings.isEmpty ? Colors.green : Colors.red,
                ),
                const SizedBox(width: 8),
                Expanded(  // 添加 Expanded
                  child: Text(
                    warnings.isEmpty ? '无异常' : '异常警告',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: warnings.isEmpty ? Colors.green : Colors.red,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            if (warnings.isNotEmpty) ...[
              const SizedBox(height: 8),
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 200),
                child: SingleChildScrollView(
                  child: Column(
                    children: warnings.map((warning) => Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Row(
                        children: [
                          const Icon(Icons.error_outline,
                              color: Colors.red,
                              size: 16
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              warning.toString(),
                              overflow: TextOverflow.ellipsis,
                              maxLines: 2,
                            ),
                          ),
                        ],
                      ),
                    )).toList(),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildAnalyticsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '实时数据',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SingleChildScrollView(  // 添加水平滚动
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _buildDataItem(
                    '总人数',
                    _analysisData?['total_count']?.toString() ?? '0',
                    Icons.people,
                  ),
                  const SizedBox(width: 16),
                  _buildDataItem(
                    '密度',
                    '${((_analysisData?['crowd_density'] ?? 0) * 100).toStringAsFixed(1)}%',
                    Icons.density_medium,
                  ),
                  const SizedBox(width: 16),
                  _buildDataItem(
                    '速度',
                    '${(_analysisData?['velocity'] ?? 0).toStringAsFixed(1)} m/s',
                    Icons.speed,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDataItem(String label, String value, IconData icon) {
    return Container(
      constraints: const BoxConstraints(minWidth: 100),  // 设置最小宽度
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 24, color: Colors.blue),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(fontSize: 14),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  // endregion

  //region 其他组件
  Widget _buildLocationSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: List.generate(
          locations.length,
              (index) => _buildTabItem(
            locations[index]['name']!,
            isSelected: _selectedVideoIndex == index,
            onTap: () => _onTabSelected(index),
          ),
        ),
      ),
    );
  }

  Widget _buildTabItem(String text, {bool isSelected = false, VoidCallback? onTap}) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            border: Border(
              bottom: BorderSide(
                color: isSelected ? Colors.red : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Text(
            text,
            style: TextStyle(
              fontSize: 16,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              color: isSelected ? Colors.red : Colors.black54,
            ),
          ),
        ),
      ),
    );
  }

  //endregion

  // region 辅助函数
  List<Offset> _getPositionsFromData() {
    final positions = _analysisData?['positions'] as List? ?? [];
    return positions.map((pos) {
      if (pos is List && pos.length >= 2) {
        return Offset(pos[0].toDouble(), pos[1].toDouble());
      }
      return const Offset(0, 0);
    }).toList();
  }
  // endregion
}
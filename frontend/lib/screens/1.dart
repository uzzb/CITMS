
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../widgets/expandable_nav_bar.dart';
import '../widgets/web_socket_video_stream_widget.dart';

class MonitoringScreen extends StatefulWidget {
  const MonitoringScreen({Key? key}) : super(key: key);

  @override
  _MonitoringScreenState createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  final GlobalKey<WebSocketVideoStreamWidgetState> _webSocketVideoStreamWidgetkey =
  GlobalKey<WebSocketVideoStreamWidgetState>();

  bool _isHistoryExpanded = false;
  int _selectedVideoIndex = 0;
  DateTime _selectedDate = DateTime.now();
  TimeOfDay _startTime = const TimeOfDay(hour: 0, minute: 0);
  TimeOfDay _endTime = const TimeOfDay(hour: 23, minute: 59);


  void _onItemSelected(int index) {
    switch (index) {
      case 1:
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/traffic_flow_analysis');
        break;
      case 3:
        Navigator.pushReplacementNamed(context, '/user_settings');
        break;
    }
  }

  void _onTabSelected(int index){
    setState(() {
      _selectedVideoIndex = index;
    });
    if(_webSocketVideoStreamWidgetkey.currentState != null){
      _webSocketVideoStreamWidgetkey.currentState!.sendVideoIndex(index);
    }else{
      print("WebSocket connection not established");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color.fromRGBO(251, 21, 21, 0.72),
        title: const Text("监控", style: TextStyle(fontWeight: FontWeight.bold)),
        automaticallyImplyLeading: false,
        elevation: 0,
      ),
      body: SafeArea(
        child: Stack(
          children: <Widget>[
            Column(
              children: <Widget>[
                const SizedBox(height: 70),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey[300]!),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    margin: const EdgeInsets.all(16),
                    child:  WebSocketVideoStreamWidget(
                      key: _webSocketVideoStreamWidgetkey,
                      onWebSocketInitialized: (state){
                        //可以在这里处理初始化后的逻辑
                      },
                    ),
                  ),
                ),
                _buildVideoTabBar(),
                _buildAnomalySection(),
                _buildHistorySection(),
              ],
            ),
            Positioned(
              left: 0,
              top: 0,
              child: ExpandableNavBar(
                selectedIndex: 1,
                onItemSelected: _onItemSelected,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoTabBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildTabItem('南三国旗杆台阶', isSelected: _selectedVideoIndex == 0, onTap: () {
            _onTabSelected(0);
          }),
          _buildTabItem('南三休息室后外', isSelected: _selectedVideoIndex == 1, onTap: () {
            _onTabSelected(1);
          }),
          _buildTabItem('南四舍与南五舍南出口', isSelected: _selectedVideoIndex == 2, onTap: () {
            _onTabSelected(2);
          }),
        ],
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

  Widget _buildAnomalySection() {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFFFE4E1),
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.red.withOpacity(0.1),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.warning, color: Colors.red, size: 24),
              SizedBox(width: 8),
              Text(
                '异常行为',
                style: TextStyle(
                  color: Colors.red,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text('无异常行为', style: TextStyle(fontSize: 16, color: Colors.black54)),
          const SizedBox(height: 8),
          _buildAnomalyRecord('2024/10/10 14:36 车辆超速'),
          _buildAnomalyRecord('2024/10/10 14:36 车辆逆行'),
        ],
      ),
    );
  }

  Widget _buildAnomalyRecord(String record) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red, size: 16),
          const SizedBox(width: 8),
          Text(
            record,
            style: const TextStyle(color: Colors.red, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildHistorySection() {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: () {
              setState(() {
                _isHistoryExpanded = !_isHistoryExpanded;
              });
            },
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Row(
                  children: [
                    Icon(Icons.history, color: Colors.blue, size: 24),
                    SizedBox(width: 8),
                    Text(
                      '历史记录',
                      style: TextStyle(
                        color: Colors.blue,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Icon(
                  _isHistoryExpanded ? Icons.expand_less : Icons.expand_more,
                  color: Colors.blue,
                ),
              ],
            ),
          ),
          if (_isHistoryExpanded) ...[
            const SizedBox(height: 16),
            _buildDatePicker(),
            const SizedBox(height: 8),
            _buildTimePicker(),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _onSearchHistory,
              child: const Text('查询历史记录'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildDatePicker() {
    return Row(
      children: [
        const Icon(Icons.calendar_today, color: Colors.grey),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            '日期: ${DateFormat('yyyy-MM-dd').format(_selectedDate)}',
            style: const TextStyle(fontSize: 16),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        TextButton(
          onPressed: () async {
            final DateTime? picked = await showDatePicker(
              context: context,
              initialDate: _selectedDate,
              firstDate: DateTime(2000),
              lastDate: DateTime.now(),
            );
            if (picked != null && picked != _selectedDate) {
              setState(() {
                _selectedDate = picked;
              });
            }
          },
          child: const Text('选择日期'),
        ),
      ],
    );
  }

  Widget _buildTimePicker() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.access_time, color: Colors.grey),
            const SizedBox(width: 8),
            const Text('时间:', style: TextStyle(fontSize: 16)),
          ],
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            Expanded(
              child: Text(
                '${_startTime.format(context)} - ${_endTime.format(context)}',
                style: const TextStyle(fontSize: 16),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            TextButton(
              onPressed: () async {
                await _selectTimeRange();
              },
              child: const Text('选择时间'),
            ),
          ],
        ),
      ],
    );
  }

  Future<void> _selectTimeRange() async {
    final TimeOfDay? pickedStart = await showTimePicker(
      context: context,
      initialTime: _startTime,
    );
    if (pickedStart != null && pickedStart != _startTime) {
      setState(() {
        _startTime = pickedStart;
      });
    }

    if (pickedStart != null) {
      final TimeOfDay? pickedEnd = await showTimePicker(
        context: context,
        initialTime: _endTime,
      );
      if (pickedEnd != null && pickedEnd != _endTime) {
        setState(() {
          _endTime = pickedEnd;
        });
      }
    }
  }

  void _onSearchHistory() {
    // 实现查询历史记录的逻辑
    print('查询历史记录：日期 ${DateFormat('yyyy-MM-dd').format(_selectedDate)}, 时间 ${_startTime.format(context)} - ${_endTime.format(context)}');
  }
}
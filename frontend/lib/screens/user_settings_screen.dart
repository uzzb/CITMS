import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../widgets/custom_app_bar.dart';
import '../widgets/expandable_nav_bar.dart';
import 'monitoring_analysis_screen.dart';

class UserSettingsScreen extends StatefulWidget {
  @override
  _UserSettingsScreenState createState() => _UserSettingsScreenState();
}

class _UserSettingsScreenState extends State<UserSettingsScreen> {
  int _selectedIndex = 2;

  void _onItemSelected(int index) {
    setState(() {
      _selectedIndex = index;
    });
    switch (index) {
      case 1:
        Navigator.push(context,
            MaterialPageRoute(builder: (context) => MonitoringAnalysisScreen()));
        break;
      case 2:
        break;
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        selectedIndex: 2,
        onItemSelected: _onItemSelected,
        title: "用户设置",
      ),
      body: Stack(
        children: [
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                SvgPicture.asset(
                  'assets/images/avatar.svg',
                  height: 100,
                ),
                const SizedBox(height: 50),
                _buildMenuItem(
                  icon: Icons.account_box,
                  text: '个人信息管理',
                  onTap: () {
                    // 添加点击事件处理
                  },
                ),
                _buildMenuItem(
                  icon: Icons.settings,
                  text: '设置',
                  onTap: () {
                    // 添加点击事件处理
                  },
                ),
                _buildMenuItem(
                  icon: Icons.notifications,
                  text: '通知',
                  onTap: () {
                    // 添加点击事件处理
                  },
                ),
                _buildMenuItem(
                  icon: Icons.help,
                  text: '帮助',
                  onTap: () {
                    // 添加点击事件处理
                  },
                ),
              ],
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              padding: const EdgeInsets.all(16),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Text(
                    '联系客服',
                    style: TextStyle(
                      fontSize: 12,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    '13491308418@wust.edu.cn',
                    style: TextStyle(
                      fontSize: 8,
                      color: Colors.blue,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMenuItem({
    required IconData icon,
    required String text,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
          border: Border(
            bottom: BorderSide(color: Colors.grey[300]!, width: 1),
          ),
        ),
        child: Row(
          children: [
            Icon(icon, size: 28, color: Colors.blue),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                text,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const Icon(Icons.chevron_right, size: 28, color: Colors.grey),
          ],
        ),
      ),
    );
  }



}

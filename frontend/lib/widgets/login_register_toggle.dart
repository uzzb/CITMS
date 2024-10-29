import 'package:flutter/material.dart';

class LoginRegisterToggle extends StatefulWidget {
  final Function(bool) onToggle;

  LoginRegisterToggle({required this.onToggle});

  @override
  _LoginRegisterToggleState createState() => _LoginRegisterToggleState();
}

class _LoginRegisterToggleState extends State<LoginRegisterToggle> {
  bool _isLoginSelected = true;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 300,
      height: 50,
      decoration: BoxDecoration(
        color: Colors.grey[300],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          _buildToggleButton('登录', _isLoginSelected, () {
            setState(() {
              _isLoginSelected = true;
              widget.onToggle(true);
            });
          }),
          _buildToggleButton('注册', !_isLoginSelected, () {
            setState(() {
              _isLoginSelected = false;
              widget.onToggle(false);
            });
          }),
        ],
      ),
    );
  }

  Widget _buildToggleButton(String text, bool isSelected, VoidCallback onTap) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            color: isSelected ? Colors.white : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Center(
            child: Text(
              text,
              style: TextStyle(
                color: isSelected ? Colors.black : Colors.black54,
                fontSize: 18,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

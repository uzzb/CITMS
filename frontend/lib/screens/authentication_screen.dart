import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:front/screens/monitoring_analysis_screen.dart';
import 'package:front/widgets/login_register_toggle.dart';

class AuthenticationScreen extends StatefulWidget {
  @override
  _AuthenticationScreenState createState() => _AuthenticationScreenState();
}

class _AuthenticationScreenState extends State<AuthenticationScreen> {
  bool _isLoginSelected = true;

  void _toggleLoginRegister(bool isLoginSelected) {
    setState(() {
      _isLoginSelected = isLoginSelected;
    });
  }

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: _isLoginSelected
            ? const Color.fromRGBO(251, 21, 21, 0.72)
            : const Color.fromRGBO(204, 204, 204, 0.80),
        title: const Text(''),
        automaticallyImplyLeading: false,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const SizedBox(height: 60),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: SvgPicture.asset(
                _isLoginSelected
                    ? 'assets/images/login.svg'
                    : 'assets/images/signUp.svg',
                key: ValueKey(_isLoginSelected ? 'login' : 'signUp'),
                height: 100,
              ),
            ),
            const SizedBox(height: 10),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: Text(
                _isLoginSelected ? '登录' : '注册',
                key: ValueKey(_isLoginSelected ? '登录' : '注册'),
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 40),
            LoginRegisterToggle(onToggle: _toggleLoginRegister),
            const SizedBox(height: 40),
            _buildInputField(),
            const SizedBox(height: 40),
            _buildInputField(),
            const SizedBox(height: 40),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => MonitoringAnalysisScreen()),
                );
              },
              child: _buildActionButton(),
            ),
            if (_isLoginSelected)
              const Padding(
                padding:  EdgeInsets.only(left: 30, top: 10),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    '忘记密码',
                    style:  TextStyle(
                    color: Colors.blue,
                    fontSize: 16,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildInputField() {
    return Container(
      width: 300,
      height: 50,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.black, width: 1),
      ),
    );
  }

  Widget _buildActionButton() {
    return Container(
      width: 300,
      height: 50,
      decoration: BoxDecoration(
        gradient: _isLoginSelected
            ? null
            : const LinearGradient(
          colors: [
            Color.fromRGBO(255, 255, 255, 0.85),
            Color.fromRGBO(153, 153, 153, 0.85),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        color: _isLoginSelected
            ? const Color.fromRGBO(251, 21, 21, 0.80)
            : null,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Center(
        child: Text(
          _isLoginSelected ? '登录' : '注册',
          style: TextStyle(
            color: _isLoginSelected
                ? const Color.fromRGBO(231, 219, 219, 1.0)
                : const Color(0xFF393838),
            fontSize: 18,
          ),
        ),
      ),
    );
  }

}

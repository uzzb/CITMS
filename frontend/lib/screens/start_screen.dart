
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:front/screens/authentication_screen.dart';

class StartScreen extends StatefulWidget{
  @override
  _StartScreenState createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen> {
  bool _isLoginSelected = true;

  void _toggleLoginRegister(bool isLoginSelected){
    setState(() {
      _isLoginSelected = isLoginSelected;
    });
  }

  @override
  void initState(){
    super.initState();

  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/images/zhang-kaiyv-rkyfaZ1vkAs-unsplash.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.end, // Aligns children at the bottom
            children: [
              GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => AuthenticationScreen()),
                  );
                },
                child: Container(
                  width: 300,
                  height: 50,
                  decoration: BoxDecoration(
                    color: Color(0x99FB1515), // Red color with 52% opacity
                    borderRadius: BorderRadius.circular(12), // Rounded corners with radius 28
                  ),
                  child: Center(
                    child: Text(
                      '登录',
                      style: TextStyle(
                        color: Color(0xFFE7DBDB), // Text color with 100% opacity
                        fontSize: 18,
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(height: 20),
              GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => AuthenticationScreen()),
                  );
                },
                child: Container(
                  width: 300,
                  height: 50,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Color.fromRGBO(255, 255, 255, 0.85), // 0% FFFFF with 52% opacity
                        Color.fromRGBO(153, 153, 153, 0.85), // 100% 999999 with 52% opacity
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(12), // Rounded corners with radius 28
                  ),
                  child: Center(
                    child: Text(
                      '注册',
                      style: TextStyle(
                        color: Color(0xFF393838), // Text color 393838 with 100% opacity
                        fontSize: 18,
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(height: 20), // Add some space at the bottom
            ],
          ),
        ),
      ),
    );
  }


}
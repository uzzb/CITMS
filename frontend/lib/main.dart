import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:front/screens/authentication_screen.dart';
import 'package:front/screens/start_screen.dart';
import 'package:front/screens/user_settings_screen.dart';
import 'screens/monitoring_analysis_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
  ));
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(primarySwatch: Colors.blue),
      home: StartScreen(),
      routes: {
        '/authentication': (context) => AuthenticationScreen(),
        '/monitoring': (context) => MonitoringAnalysisScreen(),
        '/user_settings': (context) => UserSettingsScreen(),
      },
    );
  }
}
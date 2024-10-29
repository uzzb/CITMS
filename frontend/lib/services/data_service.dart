import 'dart:async';

import 'package:front/services/api_service.dart';

class DataResult {
  final String status;
  final List<String> alerts;

  DataResult(this.status, this.alerts);
}

class DataService {
  final ApiService _apiService = ApiService();

  Future<DataResult> fetchData() async {
    final alerts = await _apiService.getAlerts();

    return DataResult('Processing', alerts);

    // await Future.delayed(Duration(seconds: 1));
    //
    // return DataResult(
    //   'Processing',
    //   ['Simulated alert at ${DateTime.now()}'],
    // );
  }

  // Future<List<String>> fetchDataAlertsFromBackend() async {
  //   await Future.delayed(Duration(seconds: 1));
  //   return ['Overspeed alert detected at 100, 150', 'Another alert at 200, 300'];
  // }
}
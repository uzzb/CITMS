import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {

  // final String baseUrl = "http://127.0.0.1:8000/";

  final String baseUrl = "http://10.0.2.2:8000/";

  Stream<List<int>> sendVideoUrlAndReceiveVideo(String videoUrl) async* {
    final url = Uri.parse("${baseUrl}video_stream/");
    final request = http.Request('POST', url)
      ..headers['Content-Type'] = 'application/json'
      ..body = json.encode({'video_url': videoUrl});

    print("Send request to $url with body: ${request.body}");

    final response = await http.Client().send(request);

    if (response.statusCode == 200) {
      yield* response.stream;
    } else {
      throw Exception('Failed to send video URL');
    }
  }

  Future<List<String>> getAlerts() async{
    final url = Uri.parse("${baseUrl}get_alerts/");
    final response = await http.get(url);

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      List<String> alerts = List<String>.from(data['alert']);
      return alerts;
    } else {
      throw Exception('Failed to get alerts');
    }
  }
}

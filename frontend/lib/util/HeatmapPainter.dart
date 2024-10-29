// 热力图绘制器
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class HeatmapPainter extends CustomPainter {
  final double density;
  final List<Offset> positions;

  HeatmapPainter({required this.density, required this.positions});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.fill;

    // 绘制背景网格
    for (int i = 0; i < size.width; i += 20) {
      for (int j = 0; j < size.height; j += 20) {
        final rect = Rect.fromLTWH(i.toDouble(), j.toDouble(), 20, 20);
        final gridPaint = Paint()
          ..style = PaintingStyle.stroke
          ..color = Colors.grey.withOpacity(0.2);
        canvas.drawRect(rect, gridPaint);
      }
    }

    // 绘制热点
    for (var position in positions) {
      final shader = RadialGradient(
        colors: [
          Colors.red.withOpacity(0.3),
          Colors.transparent,
        ],
      ).createShader(
        Rect.fromCircle(center: position, radius: 50),
      );

      canvas.drawCircle(
        position,
        50,
        Paint()..shader = shader,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

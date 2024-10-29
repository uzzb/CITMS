import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class DirectionPainter extends CustomPainter {
  final List<dynamic> directions;

  DirectionPainter({required this.directions});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final paint = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    for (var direction in directions) {
      final angle = direction['angle'] as double;
      final count = direction['count'] as int;
      final length = count * 10.0; // 根据数量调整箭头长度

      final dx = cos(angle) * length;
      final dy = sin(angle) * length;
      final end = center + Offset(dx, dy);

      // 画箭头主体
      canvas.drawLine(center, end, paint);

      // 画箭头头部
      final headLength = length * 0.2;
      final headAngle = pi / 6;
      final head1 = end - Offset(
        cos(angle + headAngle) * headLength,
        sin(angle + headAngle) * headLength,
      );
      final head2 = end - Offset(
        cos(angle - headAngle) * headLength,
        sin(angle - headAngle) * headLength,
      );

      canvas.drawLine(end, head1, paint);
      canvas.drawLine(end, head2, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
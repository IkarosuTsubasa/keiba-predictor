import 'package:flutter/material.dart';

enum NativeNavIconKind { races, history, reports, more }

class NativeNavIcon extends StatelessWidget {
  const NativeNavIcon(this.kind, {super.key});

  final NativeNavIconKind kind;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: const Size.square(24),
      painter: _NativeNavIconPainter(
        kind,
        IconTheme.of(context).color ?? Colors.black,
      ),
    );
  }
}

class _NativeNavIconPainter extends CustomPainter {
  const _NativeNavIconPainter(this.kind, this.color);

  final NativeNavIconKind kind;
  final Color color;

  Paint get _stroke => Paint()
    ..color = color
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.7
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  @override
  void paint(Canvas canvas, Size size) {
    switch (kind) {
      case NativeNavIconKind.races:
        canvas.drawOval(const Rect.fromLTRB(4.5, 8, 19.5, 20.5), _stroke);
        canvas.drawOval(
          const Rect.fromLTRB(8, 11.43, 16, 17.07),
          _stroke..strokeWidth = 1.55,
        );
        canvas.drawPath(
          Path()
            ..moveTo(12, 10)
            ..lineTo(12, 3.5)
            ..moveTo(12, 3.8)
            ..lineTo(17.25, 3.8)
            ..lineTo(16.15, 5.8)
            ..lineTo(17.25, 7.8)
            ..lineTo(12, 7.8),
          _stroke,
        );
      case NativeNavIconKind.history:
        canvas.drawCircle(const Offset(12, 12), 9, _stroke);
        canvas.drawPath(
          Path()
            ..moveTo(7.25, 13.45)
            ..lineTo(10, 10.9)
            ..lineTo(12.5, 13.25)
            ..lineTo(16.75, 8.75)
            ..lineTo(16.75, 12.25)
            ..moveTo(16.75, 8.75)
            ..lineTo(13.25, 8.75),
          _stroke..strokeWidth = 1.75,
        );
      case NativeNavIconKind.reports:
        canvas.drawPath(
          Path()
            ..moveTo(6.5, 3.5)
            ..lineTo(14.5, 3.5)
            ..lineTo(18.5, 7.5)
            ..lineTo(18.5, 20.5)
            ..lineTo(6.5, 20.5)
            ..close()
            ..moveTo(14.5, 3.5)
            ..lineTo(14.5, 7.5)
            ..lineTo(18.5, 7.5),
          _stroke,
        );
        final chart = _stroke..strokeWidth = 1.75;
        canvas.drawLine(
          const Offset(9.25, 17),
          const Offset(9.25, 14.4),
          chart,
        );
        canvas.drawLine(const Offset(12, 17), const Offset(12, 11.8), chart);
        canvas.drawLine(
          const Offset(14.75, 17),
          const Offset(14.75, 9.8),
          chart,
        );
        canvas.drawLine(const Offset(9, 17), const Offset(15, 17), chart);
      case NativeNavIconKind.more:
        for (final rect in const [
          Rect.fromLTRB(5.5, 5.5, 10.5, 10.5),
          Rect.fromLTRB(13.5, 5.5, 18.5, 10.5),
          Rect.fromLTRB(5.5, 13.5, 10.5, 18.5),
          Rect.fromLTRB(13.5, 13.5, 18.5, 18.5),
        ]) {
          canvas.drawRect(rect, _stroke);
        }
        canvas.drawPath(
          Path()
            ..moveTo(12, 11.15)
            ..lineTo(12.38, 11.62)
            ..lineTo(12.95, 11.85)
            ..lineTo(12.38, 12.08)
            ..lineTo(12, 12.55)
            ..lineTo(11.62, 12.08)
            ..lineTo(11.05, 11.85)
            ..lineTo(11.62, 11.62)
            ..close(),
          Paint()..color = color,
        );
    }
  }

  @override
  bool shouldRepaint(covariant _NativeNavIconPainter oldDelegate) {
    return kind != oldDelegate.kind || color != oldDelegate.color;
  }
}
